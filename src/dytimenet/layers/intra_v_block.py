"""Shared temporal feed-forward block for each variable."""

import torch
from torch import nn

from dytimenet.layers.intra_v_moe import IntraVMoE


class IntraVBlock(nn.Module):
    """Intra-Variable Block — Dense FFN trên chiều thời gian D (shared across C).

    use_moe=True thay FFN bằng IntraVMoE; LayerNorm và residual giữ nguyên.
    """
    def __init__(self, d_model: int, ffn_ratio: float = 2.0, dropout: float = 0.1,
                 use_moe: bool = False, num_experts: int = 4, top_k: int = 1,
                 moe_dispatch: str = "sparse"):
        super().__init__()
        d_ff = int(d_model * ffn_ratio)
        self.use_moe = use_moe
        self.norm = nn.LayerNorm(d_model)
        if use_moe:
            self.moe = IntraVMoE(d_model, d_ff, num_experts, top_k, dropout, moe_dispatch)
        else:
            self.ffn  = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, mu: torch.Tensor = None,
                log_sigma: torch.Tensor = None) -> torch.Tensor:
        """x : [B, C, D] → [B, C, D]; mu/log_sigma [B, C, 1] chỉ dùng khi use_moe."""
        residual = x
        x = self.norm(x)
        x = self.moe(x, mu, log_sigma) if self.use_moe else self.ffn(x)
        return x + residual
