"""Baseline tuyến tính chạy trong cùng pipeline với DyTimeNet để so sánh công bằng.

RLinear (Li et al., 2023): RevIN + một Linear L→S dùng chung cho mọi kênh.
DLinear (Zeng et al., AAAI 2023): MovingAvg tách trend/seasonal, mỗi phần một Linear L→S dùng chung.
"""

import torch
from torch import nn

from dytimenet.layers.decomposition import MovingAvg
from dytimenet.layers.revin import RevIN


class RLinear(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, affine_revin: bool = True, **_):
        super().__init__()
        self.revin = RevIN(num_vars, affine=affine_revin)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S]"""
        return self.revin(self.linear(self.revin(x, "norm")), "denorm")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DLinear(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, decomp_kernel: int = 25, **_):
        super().__init__()
        self.decomp = MovingAvg(kernel_size=decomp_kernel)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S]"""
        seasonal, trend = self.decomp(x)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
