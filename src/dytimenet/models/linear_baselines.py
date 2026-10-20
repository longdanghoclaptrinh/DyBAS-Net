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


class PerChannelLinear(nn.Module):
    """Linear riêng cho từng kênh (individual = True của TimeBase) trên trục cuối của x [B, C, P, in]."""

    def __init__(self, channels: int, in_features: int, out_features: int):
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(channels, out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(channels, out_features).uniform_(-bound, bound))

    def forward(self, x):
        return torch.einsum("bcpi,coi->bcpo", x, self.weight) + self.bias[None, :, None, :]


def orthogonal_loss(basis):
    """Loss trực giao của TimeBase: chuẩn Frobenius phần ngoài đường chéo của Gram EᵀE (E: [..., P, R]), lấy trung bình."""
    gram = basis.transpose(-2, -1) @ basis
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    return torch.linalg.matrix_norm(off).mean()


class _Segmented(nn.Module):
    """Phần chung của SparseTSF/TimeBase: đệm trái cho đủ N·P, cắt thành [B, C, P, N] (pha × đoạn) rồi ghép lại."""

    def __init__(self, seq_len: int, pred_len: int, period: int):
        super().__init__()
        self.seq_len, self.pred_len, self.period = seq_len, pred_len, period
        self.n_in, self.n_out = math.ceil(seq_len / period), math.ceil(pred_len / period)
        self.pad = self.n_in * period - seq_len

    def _segments(self, x):
        B, C, _ = x.shape
        if self.pad:   # đệm bằng giá trị của đoạn kế tiếp (như paper: lấy từ X_{N−1})
            x = torch.cat([x[..., self.period - self.pad:self.period], x], -1)
        return x.reshape(B, C, self.n_in, self.period).transpose(2, 3)         # [B, C, P, N]

    def _merge(self, y):
        B, C = y.shape[:2]
        return y.transpose(2, 3).reshape(B, C, -1)[..., :self.pred_len]       # [B, C, S]

    def _normalize(self, seg, period_norm):
        """Trừ mean theo pha (period norm của TimeBase, trung bình trên N đoạn) hoặc mean cả cửa sổ."""
        mean = seg.mean(-1, keepdim=True) if period_norm else seg.mean((-2, -1), keepdim=True)
        return seg - mean, mean

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
        return self._merge(self.linear(self._segments(z))) + mean


class TimeBase(_Segmented):
    """Theo code chính thức (github.com/hqh0728/TimeBase): period_norm, individual, loss trực giao trên basis.

    Khi train, self.aux_loss giữ loss trực giao của batch hiện tại (trainer nhân với orth_weight).
    """

    def __init__(self, num_vars: int, seq_len: int, pred_len: int, period: int = 24, basis_num: int = 6,
                 period_norm: bool = False, individual: bool = False, **_):
        super().__init__(seq_len, pred_len, period)
        self.period_norm = period_norm
        if individual:
            self.basis = PerChannelLinear(num_vars, self.n_in, basis_num)
            self.forecast = PerChannelLinear(num_vars, basis_num, self.n_out)
        else:
            self.basis = nn.Linear(self.n_in, basis_num)        # BasisExtract: N đoạn → R basis
            self.forecast = nn.Linear(basis_num, self.n_out)    # SegmentForecast: R basis → N′ đoạn tương lai
        self.aux_loss = None

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        seg, mean = self._normalize(self._segments(x), self.period_norm)
        e = self.basis(seg)                                                          # [B, C, P, R]
        self.aux_loss = orthogonal_loss(e) if self.training else None
        return self._merge(self.forecast(e) + mean)
