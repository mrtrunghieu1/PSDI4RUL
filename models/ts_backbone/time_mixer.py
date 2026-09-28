"""
TimeMixer for RUL Regression.

Reference: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
Forecasting" (ICLR 2024).

Architecture (faithful to the original, adapted for sequence-to-one RUL regression):
  - Multi-scale avg-pool downsampling: M scales, each 2x coarser
      L=2560, M=3 -> lengths [2560, 1280, 640, 320]
  - Per-scale RevIN normalisation
  - channel_independence=1: reshape [B, L_m, C] -> [B*C, L_m, 1]
  - TokenEmbedding (Conv1d) -> [B*C, L_m, d_model]
  - Stack of PastDecomposableMixing blocks:
      * series_decomp -> (season, trend)
      * MultiScaleSeasonMixing: bottom-up learned temporal mixing (Linear on T axis)
      * MultiScaleTrendMixing:  top-down learned temporal mixing (Linear on T axis)
      * out_cross_layer FFN (feature-wise)
  - Mean pool finest scale [B*C, L_0, d_model] -> [B, d_model] -> RULHead -> [B, 1]

Key hyperparameters:
  enc_in               (C):  2
  ts_window_len / seq_len (L): 2560
  d_model:                   128
  d_ff:                      512
  e_layers:                  3    (PDM block count)
  down_sampling_layers (M):  3
  down_sampling_window:      2    (avg-pool factor)
  moving_avg:                25   (decomp kernel size)
  decomp_method:             'moving_avg'  or 'dft_decomp'
  top_k:                     5    (for DFT decomp)
  channel_independence:      1
  dropout:                   0.1
  use_norm:                  1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ts_backbone import RULHead


# ── Decomposition helpers ──────────────────────────────────────────────────────

class moving_avg(nn.Module):
    """Moving average to extract trend. Pads both ends to preserve length."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C]"""
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Decompose x into (seasonal, trend) using moving average."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DFT_series_decomp(nn.Module):
    """FFT-based series decomposition (top-k frequencies)."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, _ = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


# ── Embedding ──────────────────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """Conv1d projection from c_in channels to d_model."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        """x: [B, L, c_in]  ->  [B, L, d_model]"""
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding_wo_pos(nn.Module):
    """Value embedding only (no positional, no temporal). Suitable for raw vibration."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        """x: [B, L, c_in]  ->  [B, L, d_model]"""
        return self.dropout(self.value_embedding(x))


# ── Normalisation (RevIN-style) ────────────────────────────────────────────────

class Normalize(nn.Module):
    """Per-channel instance normalisation with optional learnable affine (RevIN-style).

    Usage:
        x_norm = norm(x, 'norm')
        x_orig = norm(x_norm, 'denorm')
    """

    def __init__(self, num_features: int, affine: bool = True, non_norm: bool = False):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """x: [B, L, C]"""
        if self.non_norm:
            return x
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev + self.mean
        return x


# ── Multi-scale mixing modules ─────────────────────────────────────────────────

class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up seasonal mixing via learned temporal projection.

    Input: season_list[m] shape [B, d_model, L_m]  (permuted before calling)
    Mixes from fine (L_0) down to coarse (L_M) using Linear(L_i -> L_{i+1}).
    Output: list of [B, L_m, d_model]  (permuted back)
    """

    def __init__(self, configs):
        super().__init__()
        seq_len = int(getattr(configs, "seq_len", getattr(configs, "ts_window_len", 2560)))
        ds_window = int(getattr(configs, "down_sampling_window", 2))
        ds_layers = int(getattr(configs, "down_sampling_layers", 3))

        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    seq_len // (ds_window ** i),
                    seq_len // (ds_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    seq_len // (ds_window ** (i + 1)),
                    seq_len // (ds_window ** (i + 1)),
                ),
            )
            for i in range(ds_layers)
        ])

    def forward(self, season_list):
        # season_list[m]: [B, d_model, L_m]
        out_high = season_list[0]
        out_low  = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]  # [B, L_0, d_model]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)  # [B, d_model, L_{i+1}]
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))  # [B, L_{i+1}, d_model]

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down trend mixing via learned temporal projection.

    Input: trend_list[m] shape [B, d_model, L_m]  (permuted before calling)
    Mixes from coarse (L_M) up to fine (L_0) using Linear(L_{i+1} -> L_i).
    Output: list of [B, L_m, d_model]  (permuted back, finest first)
    """

    def __init__(self, configs):
        super().__init__()
        seq_len = int(getattr(configs, "seq_len", getattr(configs, "ts_window_len", 2560)))
        ds_window = int(getattr(configs, "down_sampling_window", 2))
        ds_layers = int(getattr(configs, "down_sampling_layers", 3))

        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    seq_len // (ds_window ** (i + 1)),
                    seq_len // (ds_window ** i),
                ),
                nn.GELU(),
                nn.Linear(
                    seq_len // (ds_window ** i),
                    seq_len // (ds_window ** i),
                ),
            )
            for i in reversed(range(ds_layers))
        ])

    def forward(self, trend_list):
        # trend_list[m]: [B, d_model, L_m], finest first
        trend_list_rev = trend_list.copy()
        trend_list_rev.reverse()  # coarsest first

        out_low  = trend_list_rev[0]   # [B, d_model, L_M]  (coarsest)
        out_high = trend_list_rev[1]   # [B, d_model, L_{M-1}]
        out_trend_list = [out_low.permute(0, 2, 1)]  # [B, L_M, d_model]

        for i in range(len(trend_list_rev) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)  # [B, d_model, L_{M-1-i}]
            out_high = out_high + out_high_res
            out_low  = out_high
            if i + 2 <= len(trend_list_rev) - 1:
                out_high = trend_list_rev[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))  # [B, L_{M-1-i}, d_model]

        out_trend_list.reverse()  # back to finest first
        return out_trend_list


# ── PastDecomposableMixing block ───────────────────────────────────────────────

class PastDecomposableMixing(nn.Module):
    """Core PDM block — faithful to the original TimeMixer paper.

    Operates on a list of embeddings at M+1 scales (finest first):
        x_list[m]: [B, L_m, d_model]

    Steps:
      1. series_decomp -> (season_list, trend_list), each [B, d_model, L_m] after permute
      2. [if channel_independence==0] cross_layer FFN before mixing
      3. MultiScaleSeasonMixing  (bottom-up, temporal Linear)
      4. MultiScaleTrendMixing   (top-down,  temporal Linear)
      5. out = season + trend  (both [B, L_m, d_model])
      6. out_cross_layer FFN  [if channel_independence==1]
    """

    def __init__(self, configs):
        super().__init__()
        self.down_sampling_window = int(getattr(configs, "down_sampling_window", 2))
        self.channel_independence = int(getattr(configs, "channel_independence", 1))
        d_model = int(getattr(configs, "d_model", 128))
        d_ff    = int(getattr(configs, "d_ff", 512))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(float(getattr(configs, "dropout", 0.1)))

        decomp_method = getattr(configs, "decomp_method", "moving_avg")
        if decomp_method == "moving_avg":
            self.decompsition = series_decomp(int(getattr(configs, "moving_avg", 25)))
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(int(getattr(configs, "top_k", 5)))
        else:
            raise ValueError(f"Unknown decomp_method: {decomp_method}")

        if self.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend  = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        """
        Args:
            x_list: list of [B, L_m, d_model], finest first
        Returns:
            updated x_list of same shapes
        """
        length_list = [x.shape[1] for x in x_list]

        # Decompose and (optionally) apply channel FFN
        season_list = []
        trend_list  = []
        for x in x_list:
            season, trend = self.decompsition(x)          # [B, L_m, d_model]
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend  = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))  # [B, d_model, L_m]
            trend_list.append(trend.permute(0, 2, 1))    # [B, d_model, L_m]

        # Bottom-up season mixing, top-down trend mixing
        out_season_list = self.mixing_multi_scale_season(season_list)  # [B, L_m, d_model]
        out_trend_list  = self.mixing_multi_scale_trend(trend_list)    # [B, L_m, d_model]

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend                  # [B, L_m, d_model]
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])

        return out_list


# ── Main Model ─────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """TimeMixer adapted for sequence-to-one RUL regression.

    Input:  x [B, C, L]  (raw vibration, C channels, L time steps)
    Output: [B, 1]       (normalised RUL in [0, 1])
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # Core dimensions
        self.enc_in = int(getattr(configs, "enc_in", 2))
        seq_len     = int(getattr(configs, "seq_len", getattr(configs, "ts_window_len", 2560)))
        self.seq_len = seq_len
        self.down_sampling_window  = int(getattr(configs, "down_sampling_window", 2))
        self.down_sampling_layers  = int(getattr(configs, "down_sampling_layers", 3))
        self.channel_independence  = int(getattr(configs, "channel_independence", 1))
        d_model  = int(getattr(configs, "d_model", 128))
        e_layers = int(getattr(configs, "e_layers", 3))
        dropout  = float(getattr(configs, "dropout", 0.1))
        use_norm = bool(getattr(configs, "use_norm", 1))

        # Ensure configs has seq_len for MultiScale* constructors
        if not hasattr(configs, "seq_len"):
            configs.seq_len = seq_len

        # PDM encoder
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(e_layers)
        ])

        # For pre-processing (preprocessing decomp before embedding, mirroring original)
        self.preprocess = series_decomp(int(getattr(configs, "moving_avg", 25)))

        # Embedding: channel_independent -> c_in=1, else c_in=enc_in
        emb_c_in = 1 if self.channel_independence == 1 else self.enc_in
        self.enc_embedding = DataEmbedding_wo_pos(emb_c_in, d_model, dropout)

        # Per-scale RevIN normalisation (on the original enc_in channels)
        self.normalize_layers = nn.ModuleList([
            Normalize(
                self.enc_in,
                affine=True,
                non_norm=(use_norm == 0),
            )
            for _ in range(self.down_sampling_layers + 1)
        ])

        self.rul_head = RULHead(d_model, dropout)

    # ── helpers ────────────────────────────────────────────────────────────────

    def __multi_scale_process_inputs(self, x_enc):
        """Build multi-scale list.

        Args:
            x_enc: [B, L, C]
        Returns:
            list of [B, L_m, C], finest first
        """
        down_pool = nn.AvgPool1d(self.down_sampling_window)

        x_enc_t = x_enc.permute(0, 2, 1)       # [B, C, L]
        x_enc_ori = x_enc_t

        sampling_list = [x_enc]                 # scale-0: [B, L, C]
        for _ in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)          # [B, C, L/2]
            sampling_list.append(x_enc_sampling.permute(0, 2, 1))  # [B, L/2, C]
            x_enc_ori = x_enc_sampling

        return sampling_list

    def pre_enc(self, x_list):
        """Apply preprocess decomp for channel-dependent mode (mirrors original pre_enc)."""
        if self.channel_independence == 1:
            return x_list, None
        out1, out2 = [], []
        for x in x_list:
            x1, x2 = self.preprocess(x)
            out1.append(x1)
            out2.append(x2)
        return out1, out2

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 [B, C, L]
        Returns:
            [B, 1] RUL prediction in [0, 1]
        """
        B, C, L = x.shape

        # [B, C, L] -> [B, L, C]
        x_enc = x.permute(0, 2, 1)

        # Multi-scale downsampling: list of [B, L_m, C]
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # Per-scale normalisation + channel-independence reshape
        x_list = []
        for i, x_s in enumerate(x_enc_list):
            x_s = self.normalize_layers[i](x_s, "norm")   # [B, L_m, C]
            if self.channel_independence == 1:
                # [B, L_m, C] -> [B*C, L_m, 1]
                x_s = x_s.permute(0, 2, 1).contiguous().reshape(B * C, x_s.shape[1], 1)
            x_list.append(x_s)

        # Pre-processing decomp (season branch for embedding)
        x_list_season, _ = self.pre_enc(x_list)

        # Embedding: [B(*C), L_m, c_in] -> [B(*C), L_m, d_model]
        enc_out_list = [self.enc_embedding(x_s, None) for x_s in x_list_season]

        # Past Decomposable Mixing encoder
        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list)

        # Aggregate: finest scale mean-pool -> [B, d_model]
        enc_out = enc_out_list[0]              # [B*C, L_0, d_model]  or  [B, L_0, d_model]
        if self.channel_independence == 1:
            # [B*C, L_0, d_model] -> [B, C, L_0, d_model] -> mean over C and L_0
            enc_out = enc_out.reshape(B, C, enc_out.shape[1], enc_out.shape[2])
            enc_out = enc_out.mean(dim=[1, 2])   # [B, d_model]
        else:
            enc_out = enc_out.mean(dim=1)        # [B, d_model]

        return self.rul_head(enc_out)            # [B, 1]
