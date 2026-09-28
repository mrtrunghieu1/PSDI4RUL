"""
TimeMixer++ for RUL Regression.

Extends TimeMixer with:
  1. Channel mixing gate at each scale: a learned C-dim gate mixes the C
     channels before temporal mixing, capturing H/V vibration correlations.
  2. Learnable scale weights: softmax-normalised parameters weight each scale's
     mean-pooled representation before the RUL head (instead of using only the
     finest scale).
  3. Optional DFT decomposition: top-K FFT components serve as the trend term
     instead of a moving average (set --decomp_method dft_decomp).

Key hyperparameters:
  enc_in               (C):  2
  ts_window_len        (L):  2560
  d_model:                   128
  d_ff:                      512
  e_layers:                  3
  down_sampling_layers (M):  3
  down_sampling_window:      2
  moving_avg:                25
  decomp_method:             'moving_avg' | 'dft_decomp'
  top_k:                     5    (used only for dft_decomp)
  dropout:                   0.1
  use_norm:                  1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ts_backbone import RULHead


# ── Decomposition helpers ─────────────────────────────────────────────────────

class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → trend [B, L, d_model]"""
        front = x[:, :1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return trend


class DFTDecomp(nn.Module):
    """Frequency-domain decomposition: keep top-K FFT components as trend."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        """x: [B, L, d_model] → (season, trend)"""
        L = x.shape[1]
        orig_dtype = x.dtype
        # cuFFT requires float32 for non-power-of-2 lengths (AMP uses float16)
        xf = torch.fft.rfft(x.float(), dim=1)            # [B, L//2+1, d_model]
        amp = xf.abs().mean(dim=-1)                       # [B, L//2+1]
        amp[:, 0] = 0                                     # zero DC
        _, top_idx = torch.topk(amp, min(self.top_k, amp.shape[1] - 1), dim=1)

        mask = torch.zeros_like(xf.abs())                 # [B, L//2+1, d_model]
        # Broadcast top_idx over d_model dimension
        mask.scatter_(1, top_idx.unsqueeze(-1).expand(-1, -1, xf.shape[-1]), 1.0)
        trend_f = xf * mask
        trend = torch.fft.irfft(trend_f, n=L, dim=1).to(orig_dtype)  # [B, L, d_model]
        season = x - trend
        return season, trend


class SeriesDecomp(nn.Module):
    def __init__(self, method: str, kernel_size: int, top_k: int):
        super().__init__()
        if method == "dft_decomp":
            self.decomp = DFTDecomp(top_k=top_k)
        else:
            self.decomp = MovingAvgWrapper(kernel_size)

    def forward(self, x):
        return self.decomp(x)


class MovingAvgWrapper(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.ma = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.ma(x)
        return x - trend, trend


# ── Channel Mixer ─────────────────────────────────────────────────────────────

class ChannelMixer(nn.Module):
    """Mixes C channels at a given scale using a learned gate.

    Input:  [B, L, d_model]  — already channel-merged (C embedded into d_model)
    We split d_model into C groups and apply a cross-group gate, then recombine.
    This is lightweight: one Linear(C, C) per scale.

    Since we embed C channels independently into d_model via separate linear
    heads in the embedding layer, here we operate on the embedding dimension
    with a gated mixing approach.
    """

    def __init__(self, d_model: int, n_channels: int, dropout: float):
        super().__init__()
        # Per-token channel attention: [B, L, n_channels] → [B, L, n_channels]
        self.gate = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.n_channels = n_channels
        self.d_model = d_model
        # Project d_model back and forth to expose channel dimension
        self.split = nn.Linear(d_model, n_channels)
        self.merge = nn.Linear(n_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, L, d_model]"""
        ch = self.split(x)                    # [B, L, C]
        gate = self.gate(ch)                  # [B, L, C]
        ch = ch * gate                        # channel-gated mixing
        delta = self.merge(self.dropout(ch))  # [B, L, d_model]
        return x + delta                      # residual


# ── PDM++ Block ───────────────────────────────────────────────────────────────

class PDMPPBlock(nn.Module):
    """Extended PDM block with channel mixing."""

    def __init__(self, configs, n_scales: int):
        super().__init__()
        d_model = int(getattr(configs, "d_model", 128))
        d_ff = int(getattr(configs, "d_ff", 512))
        dropout = float(getattr(configs, "dropout", 0.1))
        moving_avg = int(getattr(configs, "moving_avg", 25))
        top_k = int(getattr(configs, "top_k", 5))
        decomp_method = getattr(configs, "decomp_method", "moving_avg")
        n_channels = int(getattr(configs, "enc_in", 2))
        self.n_scales = n_scales

        self.decomps = nn.ModuleList([
            SeriesDecomp(decomp_method, moving_avg, top_k) for _ in range(n_scales)
        ])

        # Channel mixer at each scale
        self.channel_mixers = nn.ModuleList([
            ChannelMixer(d_model, n_channels, dropout) for _ in range(n_scales)
        ])

        # Temporal mixing MLP per scale
        self.temporal_mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_scales)
        ])

        # Cross-scale projection (coarser → finer)
        self.cross_scale_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_scales - 1)
        ])

        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])

    def forward(self, x_list):
        out_list = []

        for m in range(self.n_scales):
            x_m = x_list[m]                              # [B, L_m, d_model]

            # Channel mixing first
            x_m = self.channel_mixers[m](x_m)

            # Decompose and temporally mix seasonal component
            season, trend = self.decomps[m](x_m)
            season = season + self.temporal_mlp[m](season)
            x_out = self.layer_norms[m](season + trend)
            out_list.append(x_out)

        # Cross-scale mixing: coarser → finer
        for m in range(self.n_scales - 2, -1, -1):
            coarse = out_list[m + 1]
            fine = out_list[m]
            L_fine = fine.shape[1]
            coarse_up = F.interpolate(
                coarse.permute(0, 2, 1),
                size=L_fine,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            coarse_up = self.cross_scale_proj[m](coarse_up)
            out_list[m] = fine + coarse_up

        return out_list


# ── Main model ────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """TimeMixer++ adapted for sequence-to-one RUL regression."""

    def __init__(self, configs):
        super().__init__()
        self.C = int(getattr(configs, "enc_in", 2))
        self.L = int(getattr(configs, "ts_window_len", 2560))
        self.M = int(getattr(configs, "down_sampling_layers", 3))
        self.ds_window = int(getattr(configs, "down_sampling_window", 2))
        d_model = int(getattr(configs, "d_model", 128))
        e_layers = int(getattr(configs, "e_layers", 3))
        dropout = float(getattr(configs, "dropout", 0.1))
        self.use_norm = bool(getattr(configs, "use_norm", 1))
        n_scales = self.M + 1

        # Per-scale input embedding
        self.embed_layers = nn.ModuleList([
            nn.Linear(self.C, d_model) for _ in range(n_scales)
        ])

        # Stack of PDM++ blocks
        self.pdm_blocks = nn.ModuleList([
            PDMPPBlock(configs, n_scales) for _ in range(e_layers)
        ])

        # Learnable scale weights: contribution of each scale to the RUL head
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        self.rul_head = RULHead(d_model, dropout)

    def _multiscale_downsample(self, x: torch.Tensor):
        x_list = [x]
        for _ in range(self.M):
            x_prev = x_list[-1]
            x_down = F.avg_pool1d(
                x_prev.permute(0, 2, 1),
                kernel_size=self.ds_window,
                stride=self.ds_window,
                padding=0,
            ).permute(0, 2, 1)
            x_list.append(x_down)
        return x_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 [B, C, L]
        Returns:
            [B, 1] RUL prediction in [0, 1]
        """
        # ── Instance normalisation ────────────────────────────────────────────
        if self.use_norm:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-8
            x = (x - mean) / std

        x = x.permute(0, 2, 1)                           # [B, L, C]

        x_scales = self._multiscale_downsample(x)        # list [B, L_m, C]

        x_list = [self.embed_layers[m](x_scales[m]) for m in range(self.M + 1)]

        for pdm in self.pdm_blocks:
            x_list = pdm(x_list)

        # ── Learnable-weighted sum of all scale pooled features ───────────────
        scale_w = F.softmax(self.scale_weights, dim=0)   # [M+1]
        x_out = sum(
            scale_w[m] * x_list[m].mean(dim=1)
            for m in range(self.M + 1)
        )                                                 # [B, d_model]

        return self.rul_head(x_out)                      # [B, 1]
