"""DyTimeNet v2 architecture extracted without changing notebook computations."""

import torch
from torch import nn
from dytimenet.layers.revin import RevIN
from dytimenet.layers.cross_v_block import CrossVBlock
from dytimenet.layers.intra_v_block import IntraVBlock
from dytimenet.layers.decomposition import MovingAvg


class DyTimeNet(nn.Module):
    """
    DyTimeNet — Dynamic Cross-Variable Dependency Network.

    Cải tiến so với bản gốc:
      1. Series Decomposition (MovingAvg):
         - x_trend    → Linear đơn giản (trend_proj)
         - x_seasonal → pipeline CrossV + IntraV như cũ
         - Cộng 2 nhánh trước khi RevIN denorm
      2. Depthwise Conv1D Temporal Encoding:
         - Thay nn.Linear(L→D) bằng DepthwiseConv(kernel=3) + GELU + Linear(L→D)
         - Mỗi biến học ngữ cảnh cục bộ của chính nó trước khi nén
    """
    def __init__(self, num_vars: int, seq_len: int, pred_len: int,
                 d_model: int = 16, num_layers: int = 1,
                 p1: float = 0.5, p2: float = 0.5,
                 ffn_ratio: float = 2.0, hidden_ratio: float = 0.75,
                 dropout: float = 0.1, affine_revin: bool = True,
                 decomp_kernel: int = 25):
        super().__init__()
        assert d_model < seq_len, (
            f"d_model ({d_model}) phải < seq_len ({seq_len}) để downsampling."
        )

        self.num_vars   = num_vars
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.d_model    = d_model
        self.num_layers = num_layers

        # ── RevIN ────────────────────────────────────────────────────────
        self.revin = RevIN(num_vars, affine=affine_revin)

        # ── [MỚI] Series Decomposition ───────────────────────────────────
        self.decomp = MovingAvg(kernel_size=decomp_kernel)

        # ── [MỚI] Depthwise Conv1D Temporal Encoding ─────────────────────
        # groups=num_vars → depthwise: mỗi biến học cục bộ độc lập
        # Không thay đổi số channel (num_vars → num_vars) để tương thích
        # với CrossVBlock phía sau vẫn dùng num_vars
        self.temporal_conv = nn.Conv1d(
            in_channels=num_vars, out_channels=num_vars,
            kernel_size=3, padding=1, groups=num_vars, bias=False
        )
        self.temporal_act  = nn.GELU()
        self.temporal_proj = nn.Linear(seq_len, d_model)   # [B,C,L] → [B,C,D]

        # ── Cross-Variable & Intra-Variable Blocks ────────────────────────
        self.cross_v_blocks = nn.ModuleList([
            CrossVBlock(num_vars, d_model, p1, p2, hidden_ratio)
            for _ in range(num_layers)])
        self.intra_v_blocks = nn.ModuleList([
            IntraVBlock(d_model, ffn_ratio, dropout)
            for _ in range(num_layers)])

        # ── Projection heads ──────────────────────────────────────────────
        self.projection  = nn.Linear(d_model,  pred_len)  # seasonal branch
        self.trend_proj  = nn.Linear(seq_len,  pred_len)  # trend branch (Linear đơn giản)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S]"""

        # 1. Chuẩn hóa đầu vào
        x = self.revin(x, "norm")                          # [B, C, L]

        # 2. Tách trend và seasonal
        x_seasonal, x_trend = self.decomp(x)              # mỗi cái [B, C, L]

        # ── Nhánh Trend: Linear đơn giản ─────────────────────────────────
        trend_out = self.trend_proj(x_trend)               # [B, C, S]

        # ── Nhánh Seasonal: Conv1D enc → CrossV + IntraV ─────────────────
        # Depthwise Conv1D: học ngữ cảnh cục bộ theo thời gian
        s = self.temporal_conv(x_seasonal)                 # [B, C, L]
        s = self.temporal_act(s)
        s = self.temporal_proj(s)                          # [B, C, D]

        # CrossV + IntraV blocks
        for cross_v, intra_v in zip(self.cross_v_blocks, self.intra_v_blocks):
            # Preserve the notebook's outer residual in addition to CrossV's inner one.
            s = cross_v(s) + s
            s = intra_v(s)

        seasonal_out = self.projection(s)                  # [B, C, S]

        # 3. Cộng 2 nhánh TRƯỚC khi denorm (cùng thang chuẩn hóa)
        out = seasonal_out + trend_out                     # [B, C, S]

        # 4. Khôi phục thang đo gốc
        out = self.revin(out, "denorm")
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
