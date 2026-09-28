"""
PatchTST for RUL Regression.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers" (ICLR 2023).

Architecture:
  - Channel-independent: each channel processed separately through the same transformer
  - Input [B, C, L] → patch via unfold → linear embed → Transformer encoder → mean pool
  - Aggregate channels → project → RULHead → scalar RUL in [0, 1]

Key hyperparameters (from args / configs):
  ts_window_len  (L):      2560
  enc_in         (C):      2     (H + V vibration channels)
  patch_len:               64
  stride:                  32
  d_model:                 128
  n_heads:                 8
  e_layers:                3
  d_ff:                    512
  dropout:                 0.1
  use_norm:                1     (RevIN instance norm)
"""

import math

import torch
import torch.nn as nn

from models.ts_backbone import RULHead


class Model(nn.Module):
    """PatchTST adapted for sequence-to-one RUL regression."""

    def __init__(self, configs):
        super().__init__()
        self.C = int(getattr(configs, "enc_in", 2))
        self.L = int(getattr(configs, "ts_window_len", 2560))
        self.patch_len = int(getattr(configs, "patch_len", 64))
        self.stride = int(getattr(configs, "stride", 32))
        d_model = int(getattr(configs, "d_model", 128))
        n_heads = int(getattr(configs, "n_heads", 8))
        e_layers = int(getattr(configs, "e_layers", 3))
        d_ff = int(getattr(configs, "d_ff", 512))
        dropout = float(getattr(configs, "dropout", 0.1))
        self.use_norm = bool(getattr(configs, "use_norm", 1))

        # Number of patches (no padding)
        self.N_patches = (self.L - self.patch_len) // self.stride + 1

        # Patch projection (channel-independent, shared weights)
        self.patch_embed = nn.Linear(self.patch_len, d_model, bias=False)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.N_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder with Pre-LN (more stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Channel aggregation: [B, C*d_model] → [B, d_model]
        self.channel_proj = nn.Linear(self.C * d_model, d_model)

        self.rul_head = RULHead(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 [B, C, L]
        Returns:
            [B, 1] RUL prediction in [0, 1]
        """
        B, C, L = x.shape

        # ── Instance normalisation (RevIN-style, per channel per sample) ──────
        if self.use_norm:
            mean = x.mean(dim=-1, keepdim=True)        # [B, C, 1]
            std = x.std(dim=-1, keepdim=True) + 1e-8
            x = (x - mean) / std

        # ── Patching ──────────────────────────────────────────────────────────
        # x: [B, C, L]  →  merge batch & channel  →  [B*C, L]
        x = x.reshape(B * C, L)

        # unfold: [B*C, N_patches, patch_len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # ── Linear patch embedding + positional encoding ──────────────────────
        x = self.patch_embed(x) + self.pos_embed    # [B*C, N_patches, d_model]

        # ── Transformer encoding ──────────────────────────────────────────────
        x = self.encoder(x)                          # [B*C, N_patches, d_model]

        # ── Mean pool over patches ────────────────────────────────────────────
        x = x.mean(dim=1)                            # [B*C, d_model]

        # ── Aggregate channels ────────────────────────────────────────────────
        x = x.reshape(B, C * x.shape[-1])            # [B, C*d_model]
        x = self.channel_proj(x)                     # [B, d_model]

        return self.rul_head(x)                      # [B, 1]
