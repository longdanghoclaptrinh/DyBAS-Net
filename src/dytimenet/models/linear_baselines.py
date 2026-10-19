"""Baseline tuyến tính chạy trong cùng pipeline với DyTimeNet để so sánh công bằng.

RLinear (Li et al., 2023): RevIN + một Linear L→S dùng chung cho mọi kênh.
  Với cycle_len > 0 và affine_revin=False, RLinear chính là CycleNet/Linear (Lin et al., NeurIPS 2024).
DLinear (Zeng et al., AAAI 2023): MovingAvg tách trend/seasonal, mỗi phần một Linear L→S dùng chung.
"""

import torch
from torch import nn

from dytimenet.layers.decomposition import MovingAvg
from dytimenet.layers.recurrent_cycle import RecurrentCycle
from dytimenet.layers.revin import RevIN


class RLinear(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, affine_revin: bool = True,
                 cycle_len: int = 0, **_):
        super().__init__()
        self.seq_len, self.pred_len, self.cycle_len = seq_len, pred_len, cycle_len
        self.revin = RevIN(num_vars, affine=affine_revin)
        self.linear = nn.Linear(seq_len, pred_len)
        if cycle_len:
            self.cycle = RecurrentCycle(cycle_len, num_vars)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S]"""
        z = self.revin(x, "norm")
        if not self.cycle_len:
            return self.revin(self.linear(z), "denorm")
        y = self.linear(z - self.cycle(cycle_index, self.seq_len))
        y = y + self.cycle((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        return self.revin(y, "denorm")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DLinear(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, decomp_kernel: int = 25, **_):
        super().__init__()
        self.decomp = MovingAvg(kernel_size=decomp_kernel)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S] (DLinear không dùng chu kỳ)"""
        seasonal, trend = self.decomp(x)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
