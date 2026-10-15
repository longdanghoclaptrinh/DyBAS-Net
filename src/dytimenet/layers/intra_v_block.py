"""Shared temporal feed-forward block for each variable."""

import torch
from torch import nn


class IntraVBlock(nn.Module):
    """Intra-Variable Block — Dense FFN trên chiều thời gian D (shared across C)."""
    def __init__(self, d_model: int, ffn_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        d_ff = int(d_model * ffn_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, D] → [B, C, D]"""
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        return x + residual
