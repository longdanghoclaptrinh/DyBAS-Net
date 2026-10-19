"""Gộp chính xác (exact folding) DyTimeNet đã train thành mô hình suy luận ít phép tính hơn.

Chỉ dùng khi suy luận (eval). Mọi tham số đều suy ra từ mô hình gốc, không train lại.
Với từng kênh c, đặt x là input thô, σ = std(x) + ε, γ/β là affine RevIN, A là ma trận MovingAvg
(padding lặp biên ⇒ A·1 = 1), T_c là ma trận Toeplitz của Conv1d depthwise (zero padding, không bias):

  seasonal trước GELU:  T_c (I − A) z = (γ_c / σ) · T_c (I − A) x          (μ và β triệt tiêu)
                       = M_c x / σ,                M_c = γ_c T_c (I − A)
  trend + inverse RevIN: y = ρ · (Proj(h) + b_p + c_c) + K_c x,
                       ρ = σ / (γ_c + ε),  κ = γ_c / (γ_c + ε)
                       K_c = κ W_t A (I − 11ᵀ/L) + 1_S 1_Lᵀ / L
                       c_c = β_c W_t 1 + b_t − β_c 1_S

Nhờ vậy RevIN-norm, MovingAvg, phép trừ seasonal, Conv1d, Linear trend và inverse RevIN
còn lại: 1 phép std, 1 einsum (ghép [M; K]), 1 phép chia, 1 addcmul.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .dytimenet import DyTimeNet


@torch.no_grad()
def _basis_response(fn, length, channels):
    """Ma trận [C, L_out, L] của một phép tuyến tính theo từng kênh, lấy từ ảnh của vector đơn vị."""
    eye = torch.eye(length, dtype=torch.float64)
    out = fn(eye[:, None, :].expand(length, channels, length).contiguous())   # [L, C, L_out]
    return out.permute(1, 2, 0).contiguous()                                    # [C, L_out, L]


class FoldedDyTimeNet(nn.Module):
    """Bản suy luận của DyTimeNet. stages điều khiển mức gộp để làm ablation:

    fold_trend    : gộp Linear trend + inverse RevIN thành K_c (cần μ khi chưa gộp seasonal).
    fold_seasonal : gộp RevIN-norm + MovingAvg + (I − A) + Conv1d thành M_c.
    concat        : ghép M_c và K_c thành một einsum duy nhất (cần cả hai cờ trên).
    """

    def __init__(self, model: DyTimeNet, fold_trend=True, fold_seasonal=True, concat=True):
        super().__init__()
        if model.training:
            raise ValueError("Gọi model.eval() trước khi gộp")
        if model.num_layers != 1:
            raise ValueError("Hiện chỉ hỗ trợ num_layers = 1")
        self.fold_trend, self.fold_seasonal = fold_trend, fold_seasonal
        self.concat = concat and fold_trend and fold_seasonal
        self.cross_residual = model.cross_residual
        C, L, S = model.num_vars, model.seq_len, model.pred_len
        self.L, self.S = L, S
        rev = model.revin
        self.eps = rev.eps
        d64 = torch.float64
        if rev.affine:
            gamma = rev.affine_weight.detach().to(d64)
            beta = rev.affine_bias.detach().to(d64)
        else:
            gamma, beta = torch.ones(C, dtype=d64), torch.zeros(C, dtype=d64)
        inv_g = 1 / (gamma + rev.eps) if rev.affine else torch.ones(C, dtype=d64)

        decomp64 = lambda t: model.decomp(t)[1]                                  # noqa: E731
        A = _basis_response(decomp64, L, 1)[0]                                    # [L, L]
        conv = model.temporal_conv
        T = _basis_response(lambda t: F.conv1d(t, conv.weight.detach().to(d64), None,
                                               conv.stride, conv.padding, conv.dilation, conv.groups), L, C)
        I = torch.eye(L, dtype=d64)
        Wt = model.trend_proj.weight.detach().to(d64)                              # [S, L]
        bt = model.trend_proj.bias.detach().to(d64)

        M = gamma[:, None, None] * (T @ (I - A))                                   # [C, L, L]
        center = I - torch.ones(L, L, dtype=d64) / L
        K = (gamma * inv_g)[:, None, None] * (Wt @ A @ center) + torch.ones(S, L, dtype=d64) / L  # [C, S, L]
        c = beta[:, None] * Wt.sum(1)[None] + bt[None] - beta[:, None]              # [C, S]
        bp = model.projection.bias.detach().to(d64)

        f32 = lambda t: t.to(torch.float32).contiguous()                           # noqa: E731
        self.register_buffer("M", f32(M))
        self.register_buffer("K", f32(K))
        self.register_buffer("MK", f32(torch.cat([M, K], 1)))                      # [C, L+S, L]
        self.register_buffer("inv_g", f32(inv_g)[None, :, None])                   # [1, C, 1]
        self.register_buffer("out_bias", f32(bp[None] + c)[None])                  # [1, C, S] khi fold_trend
        # Tham số giữ nguyên từ mô hình gốc (dùng khi chưa gộp hoặc cho phần phi tuyến).
        self.rev_affine = rev.affine
        self.rev_gamma = rev.affine_weight if rev.affine else None
        self.rev_beta = rev.affine_bias if rev.affine else None
        self.decomp, self.temporal_conv, self.trend_proj = model.decomp, model.temporal_conv, model.trend_proj
        self.enc_w, self.enc_b = model.temporal_proj.weight, model.temporal_proj.bias
        cv, iv = model.cross_v_blocks[0], model.intra_v_blocks[0]
        self.cv_norm, self.cv_w1, self.cv_b1 = cv.norm, cv.df_linear1.weight, cv.df_linear1.bias
        self.cv_w2, self.cv_b2 = cv.df_linear2.weight, cv.df_linear2.bias
        self.iv_norm = iv.norm
        self.iv_w1, self.iv_b1 = iv.ffn[0].weight, iv.ffn[0].bias
        self.iv_w2, self.iv_b2 = iv.ffn[3].weight, iv.ffn[3].bias
        self.proj_w, self.proj_b = model.projection.weight, model.projection.bias

    def _core(self, pre):
        """GELU → Linear L→D → CrossV → IntraV → Proj (không bias); pre: [B, C, L]."""
        s = F.linear(F.gelu(pre), self.enc_w, self.enc_b)                          # [B, C, D]
        n = self.cv_norm(s)
        m = F.linear(F.gelu(F.linear(n.transpose(1, 2), self.cv_w1, self.cv_b1)), self.cv_w2, self.cv_b2)
        m = m.transpose(1, 2)
        base = n if self.cross_residual == "norm" else s
        s = m + base + (s if self.cross_residual == "double" else 0)
        s = s + F.linear(F.gelu(F.linear(self.iv_norm(s), self.iv_w1, self.iv_b1)), self.iv_w2, self.iv_b2)
        return F.linear(s, self.proj_w)                                             # [B, C, S]

    def forward(self, x):
        sigma = x.std(-1, keepdim=True, unbiased=False) + self.eps                 # [B, C, 1]
        if self.concat:
            z = torch.einsum("bcl,col->bco", x, self.MK)
            pre, trend = z[..., :self.L] / sigma, z[..., self.L:]
        else:
            if self.fold_seasonal:
                pre = torch.einsum("bcl,col->bco", x, self.M) / sigma
            else:
                mu = x.mean(-1, keepdim=True)
                xn = (x - mu) / sigma
                if self.rev_affine:
                    xn = xn * self.rev_gamma[None, :, None] + self.rev_beta[None, :, None]
                seasonal, x_trend = self.decomp(xn)
                pre = self.temporal_conv(seasonal)
            if self.fold_trend:
                trend = torch.einsum("bcl,csl->bcs", x, self.K)
            else:
                trend = None
        p = self._core(pre)
        if self.fold_trend:
            return torch.addcmul(trend, sigma * self.inv_g, p + self.out_bias)
        # Không gộp trend: đi đúng đường gốc (cần xn từ nhánh không gộp seasonal).
        if self.fold_seasonal:
            raise ValueError("fold_seasonal mà không fold_trend chưa được hỗ trợ")
        out = p + self.proj_b + self.trend_proj(x_trend)
        if self.rev_affine:
            out = (out - self.rev_beta[None, :, None]) * self.inv_g
        return out * sigma + mu


def fold(model: DyTimeNet, **stages) -> FoldedDyTimeNet:
    """Trả bản suy luận đã gộp của một DyTimeNet ở chế độ eval."""
    return FoldedDyTimeNet(model, **stages).eval()
