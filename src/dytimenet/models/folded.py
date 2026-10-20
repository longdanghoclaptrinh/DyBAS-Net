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

Có hai cách hiện thực cùng một công thức (mode):

  channel : lưu sẵn [M_c; K_c] cho từng kênh, cả phần tiền xử lý còn 1 einsum. Ít phép tính nhất, hợp khi
            C·(L+S)·L nhỏ (bộ đệm nằm trong cache, latency do số phép tính quyết định).
  shared  : M_c chỉ khác nhau qua γ_c T_c, K_c chỉ khác nhau qua hệ số κ_c, nên giữ Conv depthwise (trọng số
            nhân sẵn γ_c) và một Linear trend dùng chung W_t, rồi nhân κ_c sau. Bộ nhớ O(L·S) như DLinear thay
            vì O(C·L·(L+S)); hợp khi nhiều kênh và L dài (ví dụ C = 321, L = S = 720: 2.5 GB → 2 MB).
  auto    : chọn shared khi bộ đệm của channel vượt AUTO_SHARED_BYTES.

Vì M_c 1 = 0 và K_c 1 = 1, ta nhân với x − μ thay cho x rồi cộng μ vào nhánh trend (kết quả không đổi về
toán học). Không trừ μ trước thì với cửa sổ gần hằng số (σ ≈ ε, ví dụ lượng mưa = 0 trên Weather), M_c x
triệt tiêu trong float32 rồi bị khuếch đại 1/σ, và CrossV lan sai số sang các kênh khác. Chế độ shared không
có ma trận centering nên trừ mean thêm một lần nữa cho cùng tác dụng.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .dytimenet import DyTimeNet

AUTO_SHARED_BYTES = 16 * 2**20   # ≈ L3 cache; xem results/bench_fold_modes.md


@torch.no_grad()
def _basis_response(fn, length, channels):
    """Ma trận [C, L_out, L] của một phép tuyến tính theo từng kênh, lấy từ ảnh của vector đơn vị."""
    eye = torch.eye(length, dtype=torch.float64)
    out = fn(eye[:, None, :].expand(length, channels, length).contiguous())   # [L, C, L_out]
    return out.permute(1, 2, 0).contiguous()                                    # [C, L_out, L]


def _f32(t):
    return t.to(torch.float32).contiguous()


class FoldedDyTimeNet(nn.Module):
    """Bản suy luận của DyTimeNet. stages điều khiển mức gộp để làm ablation (chỉ ở mode channel):

    fold_trend    : gộp Linear trend + inverse RevIN thành K_c (cần μ khi chưa gộp seasonal).
    fold_seasonal : gộp RevIN-norm + MovingAvg + (I − A) + Conv1d thành M_c.
    concat        : ghép M_c và K_c thành một einsum duy nhất (cần cả hai cờ trên).
    """

    def __init__(self, model: DyTimeNet, fold_trend=True, fold_seasonal=True, concat=True, mode="auto"):
        super().__init__()
        if model.training:
            raise ValueError("Gọi model.eval() trước khi gộp")
        if model.num_layers != 1:
            raise ValueError("Hiện chỉ hỗ trợ num_layers = 1")
        if mode not in ("auto", "channel", "shared"):
            raise ValueError("mode phải là auto, channel hoặc shared")
        self.fold_trend, self.fold_seasonal = fold_trend, fold_seasonal
        self.concat = concat and fold_trend and fold_seasonal
        self.cross_residual = model.cross_residual
        C, L, S = model.num_vars, model.seq_len, model.pred_len
        self.L, self.S = L, S
        self.cycle_len = model.cycle_len
        if mode == "auto":
            mode = "shared" if 4 * C * (L + S) * L > AUTO_SHARED_BYTES and self.concat else "channel"
        if mode == "shared" and not self.concat:
            raise ValueError("Chế độ shared chỉ hỗ trợ gộp đầy đủ")
        self.mode = mode
        rev = model.revin
        self.eps = rev.eps
        d64 = torch.float64
        if rev.affine:
            gamma = rev.affine_weight.detach().to(d64)
            beta = rev.affine_bias.detach().to(d64)
        else:
            gamma, beta = torch.ones(C, dtype=d64), torch.zeros(C, dtype=d64)
        inv_g = 1 / (gamma + rev.eps) if rev.affine else torch.ones(C, dtype=d64)
        self.register_buffer("inv_g", _f32(inv_g)[None, :, None])                  # [1, C, 1]
        if mode == "shared":
            self._build_shared(model, gamma, beta, inv_g)
        else:
            self._build_channel(model, gamma, beta, inv_g)

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

    def _cycle_tables(self, model, beta, Wt, bt, bp, seasonal_map):
        """Bảng tra theo pha φ cho RCF: e_φ = T(β1 − q_φ) [W, C, L], g_φ = b_p + W_t(β1 − q_φ) + b_t + q′_φ − β1."""
        C, L, S, W = model.num_vars, self.L, self.S, self.cycle_len
        Q = model.cycle.data.detach().to(torch.float64)                            # [W, C]
        phase = torch.arange(W)[:, None]
        q_in = Q[(phase + torch.arange(L)[None]) % W].permute(0, 2, 1)             # [W, C, L]
        q_out = Q[(phase + L + torch.arange(S)[None]) % W].permute(0, 2, 1)        # [W, C, S]
        v = beta[None, :, None] - q_in                                              # [W, C, L]
        g = bp + torch.einsum("sl,wcl->wcs", Wt, v) + bt + q_out - beta[None, :, None]
        self.register_buffer("pre_table", _f32(seasonal_map(v)))
        self.register_buffer("out_table", _f32(g))

    def _build_channel(self, model, gamma, beta, inv_g):
        C, L, S = model.num_vars, self.L, self.S
        d64 = torch.float64
        A = _basis_response(lambda t: model.decomp(t)[1], L, 1)[0]                 # [L, L]
        conv = model.temporal_conv
        T = _basis_response(lambda t: F.conv1d(t, conv.weight.detach().to(d64), None,
                                               conv.stride, conv.padding, conv.dilation, conv.groups), L, C)
        I = torch.eye(L, dtype=d64)
        Wt = model.trend_proj.weight.detach().to(d64)                              # [S, L]
        bt = model.trend_proj.bias.detach().to(d64)
        center = I - torch.ones(L, L, dtype=d64) / L
        bp = model.projection.bias.detach().to(d64)
        if self.cycle_len:
            # RCF: seasonal = trend = r = z − q_φ (không có MovingAvg). Không còn (I − A) để triệt tiêu μ,
            # nên dùng ma trận centering; β và chu kỳ thành bảng tra theo pha φ:
            #   pre = γ T C x / σ + e_φ,        e_φ = T (β1 − q_φ)
            #   ŷ   = K x + ρ (Proj(h) + g_φ),  K = κ W_t C + 11ᵀ/L
            #   g_φ = b_p + W_t (β1 − q_φ) + b_t + q′_φ − β1      (q′_φ: chu kỳ của tương lai)
            if not self.concat:
                raise ValueError("Chế độ RCF chỉ hỗ trợ gộp đầy đủ")
            M = gamma[:, None, None] * (T @ center)
            K = (gamma * inv_g)[:, None, None] * (Wt @ center)[None] + torch.ones(S, L, dtype=d64) / L
            self._cycle_tables(model, beta, Wt, bt, bp, lambda v: torch.einsum("cij,wcj->wci", T, v))
            c = torch.zeros(C, S, dtype=d64)
        else:
            M = gamma[:, None, None] * (T @ (I - A))                               # [C, L, L]
            K = (gamma * inv_g)[:, None, None] * (Wt @ A @ center) + torch.ones(S, L, dtype=d64) / L  # [C, S, L]
            c = beta[:, None] * Wt.sum(1)[None] + bt[None] - beta[:, None]          # [C, S]
        self.register_buffer("M", _f32(M))
        self.register_buffer("K", _f32(K))
        self.register_buffer("MK", _f32(torch.cat([M, K], 1)))                     # [C, L+S, L]
        self.register_buffer("out_bias", _f32(bp[None] + c)[None])                 # [1, C, S] khi fold_trend

    def _build_shared(self, model, gamma, beta, inv_g):
        d64 = torch.float64
        conv = model.temporal_conv
        w = conv.weight.detach().to(d64)                                            # [C, 1, k]
        self.conv_groups, self.conv_pad = conv.groups, conv.padding[0]
        Wt = model.trend_proj.weight.detach().to(d64)                              # [S, L]
        bt = model.trend_proj.bias.detach().to(d64)
        bp = model.projection.bias.detach().to(d64)
        self.register_buffer("conv_w", _f32(gamma[:, None, None] * w))             # γ_c nhân sẵn vào kernel
        self.register_buffer("trend_w", _f32(Wt))
        self.register_buffer("kappa", _f32(gamma * inv_g)[None, :, None])          # [1, C, 1]
        if self.cycle_len:
            self._cycle_tables(model, beta, Wt, bt, bp,
                               lambda v: F.conv1d(v, w, None, 1, self.conv_pad, 1, self.conv_groups))
        else:
            c = beta[:, None] * Wt.sum(1)[None] + bt[None] - beta[:, None]          # [C, S]
            self.register_buffer("out_bias", _f32(bp[None] + c)[None])

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

    def _forward_shared(self, x, cycle_index):
        sigma, mu = torch.std_mean(x, -1, keepdim=True, unbiased=False)            # [B, C, 1]
        sigma = sigma + self.eps
        xc = x - mu
        xc = xc - xc.mean(-1, keepdim=True)          # lần 2: cửa sổ hằng số thành đúng 0 (xem docstring)
        if self.cycle_len:
            seasonal, trend_in, add = xc, xc, self.out_table[cycle_index]
        else:
            seasonal, trend_in = self.decomp(xc)
            add = self.out_bias
        pre = F.conv1d(seasonal, self.conv_w, None, 1, self.conv_pad, 1, self.conv_groups) / sigma
        if self.cycle_len:
            pre = pre + self.pre_table[cycle_index]
        p = self._core(pre)
        trend = torch.addcmul(mu, self.kappa, F.linear(trend_in, self.trend_w))    # μ + κ_c W_t (A) x̃
        return torch.addcmul(trend, sigma * self.inv_g, p + add)

    def forward(self, x, cycle_index=None):
        if self.mode == "shared":
            return self._forward_shared(x, cycle_index)
        sigma, mu = torch.std_mean(x, -1, keepdim=True, unbiased=False)            # [B, C, 1]
        sigma = sigma + self.eps
        xc = x - mu
        if self.cycle_len:
            # RCF đã gộp: 1 einsum + 2 lần tra bảng theo pha.
            z = torch.einsum("bcl,col->bco", xc, self.MK)
            pre = z[..., :self.L] / sigma + self.pre_table[cycle_index]
            p = self._core(pre)
            return torch.addcmul(z[..., self.L:] + mu, sigma * self.inv_g, p + self.out_table[cycle_index])
        if self.concat:
            z = torch.einsum("bcl,col->bco", xc, self.MK)
            pre, trend = z[..., :self.L] / sigma, z[..., self.L:] + mu
        else:
            if self.fold_seasonal:
                pre = torch.einsum("bcl,col->bco", xc, self.M) / sigma
            else:
                xn = xc / sigma
                if self.rev_affine:
                    xn = xn * self.rev_gamma[None, :, None] + self.rev_beta[None, :, None]
                seasonal, x_trend = self.decomp(xn)
                pre = self.temporal_conv(seasonal)
            if self.fold_trend:
                trend = torch.einsum("bcl,csl->bcs", xc, self.K) + mu
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
    """Trả bản suy luận đã gộp của một DyTimeNet ở chế độ eval (mode mặc định: auto)."""
    return FoldedDyTimeNet(model, **stages).eval()
