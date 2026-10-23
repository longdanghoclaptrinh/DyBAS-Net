"""Baseline dùng chu kỳ học được, chạy trong cùng pipeline với DyTimeBase để so sánh công bằng.

Cài lại theo mã nguồn chính thức (github.com/ACAT-SCUT/CycleNet, github.com/ACAT-SCUT/TQNet):
CycleNet (Lin et al., NeurIPS 2024): chuẩn hóa theo cửa sổ, trừ chu kỳ học được Q ∈ R^{W×C} (RCF), Linear hoặc
  MLP L→D→S dùng chung mọi kênh, cộng lại chu kỳ của tương lai.
TQNet (Lin et al., ICML 2025): chuỗi truy vấn theo thời gian (Temporal Query, cũng là Q ∈ R^{W×C} trượt theo pha) làm
  query, chuỗi đầu vào làm key/value của một lớp attention trên trục kênh (mỗi kênh là một token L chiều), rồi MLP.

Chuẩn hóa giống code chính thức: phương sai không chệch (torch.var) + 1e-5 nằm trong căn, không có affine.
Code chính thức lấy pha tại s_end (bước dự báo đầu tiên) còn dataset ở đây trả pha tại s_begin; vì Q khởi tạo 0 và
Adam cập nhật từng phần tử, hai cách chỉ khác nhau một phép dịch vòng của Q nên quá trình huấn luyện tương đương.
"""

import torch
from torch import nn

from dytimenet.layers.recurrent_cycle import RecurrentCycle


def _normalize(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True) + 1e-5
    return (x - mean) / var.sqrt(), mean, var


class CycleNet(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, cycle_len: int = 24, d_model: int = 512,
                 mlp: bool = False, **_):
        super().__init__()
        self.seq_len, self.pred_len, self.cycle_len = seq_len, pred_len, cycle_len
        self.cycle = RecurrentCycle(cycle_len, num_vars)
        if mlp:
            self.model = nn.Sequential(nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, pred_len))
        else:
            self.model = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        """x : [B, C, L], cycle_index [B] (pha của bước đầu cửa sổ) → [B, C, S]"""
        x, mean, var = _normalize(x)
        y = self.model(x - self.cycle(cycle_index, self.seq_len))
        y = y + self.cycle((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        return y * var.sqrt() + mean

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TQNet(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int, cycle_len: int = 24, d_model: int = 512,
                 dropout: float = 0.5, **_):
        super().__init__()
        self.seq_len = seq_len
        self.query = RecurrentCycle(cycle_len, num_vars)            # Temporal Query
        self.aggregator = nn.MultiheadAttention(seq_len, num_heads=4, batch_first=True, dropout=0.5)
        self.input_proj = nn.Linear(seq_len, d_model)
        self.model = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
        self.output_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        """x : [B, C, L], cycle_index [B] → [B, C, S]"""
        x, mean, var = _normalize(x)
        query = self.query(cycle_index, self.seq_len)                                # [B, C, L]
        channel_info = self.aggregator(query=query, key=x, value=x)[0]
        h =self.input_proj(x + channel_info)                                        # [B, C, D]
        y = self.output_proj(self.model(h) + h)
        return y * var.sqrt() + mean

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
