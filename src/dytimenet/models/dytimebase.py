"""DyTimeBase: dự báo theo basis của TimeBase + trộn kênh của DyTimeNet trên không gian basis.

Ý tưởng (đọc cùng TimeBase, ICML 2025, và DyTimeNet):
  1. Trừ mean (và chu kỳ RCF nếu cycle_len > 0), cắt chuỗi dài L thành N đoạn dài P: X ∈ R^{P×N}.
  2. Rút R basis theo thời gian bằng Linear N→R dùng chung mọi kênh: E ∈ R^{P×R} (R ≪ N).
  3. Nhánh trộn kênh chạy trên E (P·R số mỗi kênh, thay vì L điểm): Linear P·R→D → CrossV (trộn kênh) →
     IntraV (FFN) → Linear D→P·R, cộng vào E. Lớp cuối khởi tạo 0 nên lúc đầu mô hình đúng bằng TimeBase.
  4. Dự báo theo đoạn bằng Linear R→N′ dùng chung, ghép lại thành S điểm, cộng mean (và chu kỳ tương lai).

Vì sao nhanh: mọi phép trên trục thời gian đều chạy trên N, R, P (vài chục) thay vì L, S (hàng trăm); phần trộn kênh
chạy trên vector D chiều. Ví dụ Electricity (C = 321, L = S = 720): khoảng 20M MACs so với 211M của DyTimeNet và
333M của DLinear.
"""

import torch
from torch import nn

from dytimenet.layers.cross_v_block import CrossVBlock
from dytimenet.layers.intra_v_block import IntraVBlock
from dytimenet.layers.recurrent_cycle import RecurrentCycle

from .linear_baselines import PerChannelLinear, _Segmented, orthogonal_loss


class DyTimeBase(_Segmented):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, period: int = 24, basis_num: int = 6,
                 d_model: int = 64, p1: float = 0.1, p2: float = 0.1, hidden_ratio: float = 0.75,
                 ffn_ratio: float = 2.0, dropout: float = 0.1, cycle_len: int = 0, use_mix: bool = True,
                 period_norm: bool = False, individual: bool = False, **_):
        super().__init__(seq_len, pred_len, period)
        self.cycle_len, self.period_norm = cycle_len, period_norm
        if individual:   # backbone giống TimeBase individual = True
            self.basis = PerChannelLinear(num_vars, self.n_in, basis_num)
            self.forecast = PerChannelLinear(num_vars, basis_num, self.n_out)
        else:
            self.basis = nn.Linear(self.n_in, basis_num)        # BasisExtract: N đoạn → R basis
            self.forecast = nn.Linear(basis_num, self.n_out)    # SegmentForecast: R basis → N′ đoạn
        self.aux_loss = None
        self.use_mix = use_mix
        if use_mix:
            feat = period * basis_num
            self.embed = nn.Linear(feat, d_model)
            self.cross = CrossVBlock(num_vars, d_model, p1=p1, p2=p2, hidden_ratio=hidden_ratio,
                                     residual_from_norm=True)
            self.intra = IntraVBlock(d_model, ffn_ratio=ffn_ratio, dropout=dropout)
            self.out = nn.Linear(d_model, feat)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        if cycle_len:
            self.cycle = RecurrentCycle(cycle_len, num_vars)

    def mix_parameters(self):
        """Tham số của nhánh trộn kênh (để dùng lr riêng); phần còn lại là backbone TimeBase."""
        return [p for name, p in self.named_parameters() if name.split(".")[0] in ("embed", "cross", "intra", "out")]

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S]; cycle_index [B] chỉ cần khi cycle_len > 0."""
        B, C, _ = x.shape
        if self.cycle_len:
            x = x - self.cycle(cycle_index, self.seq_len)
        seg, mean = self._normalize(self._segments(x), self.period_norm)            # [B, C, P, N]
        e = self.basis(seg)                                                           # [B, C, P, R]
        self.aux_loss = orthogonal_loss(e) if self.training else None
        if self.use_mix:
            h = self.embed(e.reshape(B, C, -1))                                       # [B, C, D]
            h = self.intra(self.cross(h))
            e = e + self.out(h).reshape(e.shape)
        y = self._merge(self.forecast(e) + mean)                                      # [B, C, S]
        if self.cycle_len:
            y = y + self.cycle((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        return y
