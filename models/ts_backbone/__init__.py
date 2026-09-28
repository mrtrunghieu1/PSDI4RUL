"""
Shared components for SOTA time-series RUL backbone models.
"""

import torch.nn as nn


class RULHead(nn.Module):
    """Shared regression head: [B, d_model] → [B, 1].

    Architecture: LayerNorm → Dropout → Linear(d→d//2) → GELU → Dropout → Linear(d//2→1) → Sigmoid

    Sigmoid output keeps predictions in [0, 1], matching the piecewise RUL label range.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
