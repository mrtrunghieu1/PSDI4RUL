"""
TimesNet for RUL Regression.

Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series
Analysis" (ICLR 2023).

Architecture:
  - Input projection: [B, L, C] → [B, L, d_model]
  - Stack of TimesBlocks, each:
      1. FFT on time axis → detect top-K dominant periods
      2. Reshape 1-D sequence to 2-D [B, d, T, period]
      3. 2-D Inception convolution (reuses layers/Conv_Blocks.Inception_Block_V1)
      4. Reshape back → amplitude-weighted residual sum
  - Global mean pool → [B, d_model] → RULHead → [B, 1]

Key hyperparameters:
  enc_in        (C):  2
  ts_window_len (L):  2560
  d_model:            128
  d_ff:               512
  e_layers:           3
  top_k:              5
  num_kernels:        6
  dropout:            0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from models.ts_backbone import RULHead


class TimesBlock(nn.Module):
    """Single TimesNet block.

    Detects top-K dominant periods via FFT, folds the 1-D sequence into 2-D
    feature maps, applies 2-D Inception convolutions, then unfolds back and
    aggregates with FFT-amplitude weights.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(getattr(configs, "ts_window_len", 2560))
        self.top_k = int(getattr(configs, "top_k", 5))
        d_model = int(getattr(configs, "d_model", 128))
        d_ff = int(getattr(configs, "d_ff", 512))
        num_kernels = int(getattr(configs, "num_kernels", 6))

        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        B, L, D = x.shape

        # ── FFT-based period detection ────────────────────────────────────────
        # cuFFT requires float32 for non-power-of-2 lengths (AMP uses float16)
        xf = torch.fft.rfft(x.float(), dim=1)            # [B, L//2+1, D]
        freq_amp = xf.abs().mean(dim=-1)                  # [B, L//2+1]
        freq_amp[:, 0] = 0                                # zero DC component

        # top-K dominant frequencies (indices)
        _, top_idx = torch.topk(freq_amp, self.top_k, dim=1)   # [B, top_k]
        # Convert frequency index to period length (avoid div-by-zero at idx=0)
        top_idx_safe = top_idx.clamp(min=1)
        period_list = (L / top_idx_safe.float()).round().long()  # [B, top_k]

        # ── Per-period 2-D convolution ────────────────────────────────────────
        res_list = []
        for i in range(self.top_k):
            # Use period of the first sample (consistent within a batch)
            period = int(period_list[0, i].item())
            period = max(period, 1)

            # Pad length to multiple of period
            if L % period != 0:
                pad_len = period - (L % period)
                x_pad = F.pad(x.permute(0, 2, 1), (0, pad_len))  # [B, D, L+pad]
            else:
                x_pad = x.permute(0, 2, 1)                        # [B, D, L]

            T = x_pad.shape[-1] // period                          # freq steps
            x_2d = x_pad.reshape(B, D, T, period)                 # [B, D, T, period]

            # 2-D Inception conv
            out_2d = self.conv(x_2d)                              # [B, D, T, period]

            # Reshape back to 1-D and trim to original length
            out_1d = out_2d.reshape(B, D, -1)[:, :, :L]          # [B, D, L]
            res_list.append(out_1d.permute(0, 2, 1))             # [B, L, D]

        # ── Amplitude-weighted aggregation ────────────────────────────────────
        # Gather amplitudes for the selected top-k frequencies: [B, top_k]
        amp_weights = torch.gather(freq_amp, 1, top_idx)          # [B, top_k]
        amp_weights = F.softmax(amp_weights, dim=1)               # [B, top_k]

        out = torch.zeros_like(x)
        for i in range(self.top_k):
            w = amp_weights[:, i].unsqueeze(1).unsqueeze(2)       # [B, 1, 1]
            out = out + w * res_list[i]

        return out


class Model(nn.Module):
    """TimesNet adapted for sequence-to-one RUL regression."""

    def __init__(self, configs):
        super().__init__()
        self.C = int(getattr(configs, "enc_in", 2))
        d_model = int(getattr(configs, "d_model", 128))
        e_layers = int(getattr(configs, "e_layers", 3))
        dropout = float(getattr(configs, "dropout", 0.1))
        self.use_norm = bool(getattr(configs, "use_norm", 1))

        # Input projection: [B, L, C] → [B, L, d_model]
        self.input_proj = nn.Linear(self.C, d_model)

        self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(e_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])

        self.rul_head = RULHead(d_model, dropout)

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

        # ── TimesNet expects [B, L, C] ────────────────────────────────────────
        x = x.permute(0, 2, 1)                  # [B, L, C]
        x = self.input_proj(x)                  # [B, L, d_model]

        # ── Stack of TimesBlocks with residual + LayerNorm ────────────────────
        for block, ln in zip(self.blocks, self.layer_norms):
            x = ln(x + block(x))                # [B, L, d_model]

        # ── Global mean pool → RUL head ───────────────────────────────────────
        x = x.mean(dim=1)                        # [B, d_model]
        return self.rul_head(x)                  # [B, 1]
