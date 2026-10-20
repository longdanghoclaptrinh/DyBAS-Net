"""Baseline tuyến tính chạy trong cùng pipeline với DyTimeNet để so sánh công bằng.

RLinear (Li et al., 2023): RevIN + một Linear L→S dùng chung cho mọi kênh.
  Với cycle_len > 0 và affine_revin=False, RLinear chính là CycleNet/Linear (Lin et al., NeurIPS 2024).
DLinear (Zeng et al., AAAI 2023): MovingAvg tách trend/seasonal, mỗi phần một Linear L→S dùng chung.
SparseTSF (Lin et al., ICML 2024): trừ mean, Conv1d tổng hợp chu kỳ, cắt thành đoạn dài P, Linear N→N′ theo pha.
TimeBase (Huang et al., ICML 2025): trừ mean, cắt thành N đoạn dài P, Linear N→R (rút basis), Linear R→N′ (dự báo
  theo đoạn). Chưa cài loss trực giao (λ_orth = 0, có trong lưới tìm kiếm của paper).
"""

import math

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


class _Segmented(nn.Module):
    """Phần chung của SparseTSF/TimeBase: trừ mean, đệm trái cho đủ N·P, cắt [B·C, P, N] rồi ghép lại."""

    def __init__(self, seq_len: int, pred_len: int, period: int):
        super().__init__()
        self.seq_len, self.pred_len, self.period = seq_len, pred_len, period
        self.n_in, self.n_out = math.ceil(seq_len / period), math.ceil(pred_len / period)
        self.pad = self.n_in * period - seq_len

    def _segments(self, x):
        B, C, _ = x.shape
        if self.pad:   # đệm bằng giá trị của đoạn kế tiếp (như paper: lấy từ X_{N−1})
            x = torch.cat([x[..., self.period - self.pad:self.period], x], -1)
        return x.reshape(B * C, self.n_in, self.period).transpose(1, 2)        # [B·C, P, N]

    def _merge(self, y, B, C):
        return y.transpose(1, 2).reshape(B, C, -1)[..., :self.pred_len]       # [B, C, S]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SparseTSF(_Segmented):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, period: int = 24, **_):
        super().__init__(seq_len, pred_len, period)
        self.conv = nn.Conv1d(1, 1, 1 + 2 * (period // 2), padding=period // 2, bias=False)
        self.linear = nn.Linear(self.n_in, self.n_out, bias=False)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        B, C, L = x.shape
        mean = x.mean(-1, keepdim=True)
        z = x - mean
        z = self.conv(z.reshape(-1, 1, L)).reshape(B, C, L) + z
        return self._merge(self.linear(self._segments(z)), B, C) + mean


class TimeBase(_Segmented):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, period: int = 24, basis_num: int = 6, **_):
        super().__init__(seq_len, pred_len, period)
        self.basis = nn.Linear(self.n_in, basis_num)        # BasisExtract: N đoạn → R basis
        self.forecast = nn.Linear(basis_num, self.n_out)    # SegmentForecast: R basis → N′ đoạn tương lai

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        B, C, _ = x.shape
        mean = x.mean(-1, keepdim=True)
        return self._merge(self.forecast(self.basis(self._segments(x - mean))), B, C) + mean
