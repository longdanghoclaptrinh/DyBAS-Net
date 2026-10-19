"""DyTimeNet v2 architecture extracted without changing notebook computations."""

import torch
from torch import nn
from dytimenet.layers.revin import RevIN
from dytimenet.layers.cross_v_block import CrossVBlock
from dytimenet.layers.intra_v_block import IntraVBlock
from dytimenet.layers.decomposition import MovingAvg
from dytimenet.layers.recurrent_cycle import RecurrentCycle


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
                 decomp_kernel: int = 25, cross_residual: str = "double", cycle_len: int = 0):
        super().__init__()
        assert d_model < seq_len, (
            f"d_model ({d_model}) phải < seq_len ({seq_len}) để downsampling."
        )

        self.num_vars   = num_vars
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.d_model    = d_model
        self.num_layers = num_layers
        # "double": cross_v(s) + s như notebook (2s + f(LN s)); "single": s + f(LN s) (Fig. 5 paper);
        # "norm": LN(s) + f(LN s) (eq. 2.14 đồ án, eq. 25 paper).
        if cross_residual not in ("double", "single", "norm"):
            raise ValueError("cross_residual must be double, single, or norm")
        self.cross_residual = cross_residual
        # cycle_len > 0: RCF thay MovingAvg. Phần dư r = z − chu kỳ đi vào cả nhánh tuyến tính
        # (trend_proj) lẫn nhánh phi tuyến; chu kỳ của tương lai được cộng lại trước inverse RevIN.
        self.cycle_len = cycle_len
        if cycle_len:
            self.cycle = RecurrentCycle(cycle_len, num_vars)

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
            CrossVBlock(num_vars, d_model, p1, p2, hidden_ratio,
                        residual_from_norm=cross_residual == "norm")
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

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None) -> torch.Tensor:
        """x : [B, C, L] → [B, C, S]; cycle_index [B] chỉ cần khi cycle_len > 0."""

        # 1. Chuẩn hóa đầu vào
        x = self.revin(x, "norm")                          # [B, C, L]

        # 2. Tách trend và seasonal (MovingAvg), hoặc trừ chu kỳ học được (RCF)
        if self.cycle_len:
            if cycle_index is None:
                raise ValueError("cycle_len > 0 cần cycle_index")
            x_seasonal = x_trend = x - self.cycle(cycle_index, self.seq_len)
        else:
            x_seasonal, x_trend = self.decomp(x)          # mỗi cái [B, C, L]

        # ── Nhánh Trend: Linear đơn giản ─────────────────────────────────
        trend_out = self.trend_proj(x_trend)               # [B, C, S]

        # ── Nhánh Seasonal: Conv1D enc → CrossV + IntraV ─────────────────
        # Depthwise Conv1D: học ngữ cảnh cục bộ theo thời gian
        s = self.temporal_conv(x_seasonal)                 # [B, C, L]
        s = self.temporal_act(s)
        s = self.temporal_proj(s)                          # [B, C, D]

        # CrossV + IntraV blocks
        for cross_v, intra_v in zip(self.cross_v_blocks, self.intra_v_blocks):
            # "double" giữ residual ngoài của notebook, cộng thêm vào residual bên trong CrossV.
            s = cross_v(s) + s if self.cross_residual == "double" else cross_v(s)
            s = intra_v(s)

        seasonal_out = self.projection(s)                  # [B, C, S]

        # 3. Cộng 2 nhánh TRƯỚC khi denorm (cùng thang chuẩn hóa)
        out = seasonal_out + trend_out                     # [B, C, S]
        if self.cycle_len:
            out = out + self.cycle((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # 4. Khôi phục thang đo gốc
        out = self.revin(out, "denorm")
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
