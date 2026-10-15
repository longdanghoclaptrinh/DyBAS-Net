"""Top-k Mixture-of-Experts thay cho FFN của IntraVBlock (token = một cặp mẫu × kênh)."""

import torch
from torch import nn
from torch.nn import functional as F


def _ffn(d_model: int, d_ff: int, dropout: float) -> nn.Sequential:
    """Đúng cấu trúc FFN cũ của IntraVBlock, chỉ khác độ rộng ẩn."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )


class IntraVMoE(nn.Module):
    """
    Shared expert (d_ff/2, luôn bật) + E expert được route (mỗi cái d_ff/(2k)).

    Router: Linear(D + 2 → E) trên [token ; μ ; log σ], μ/σ lấy từ RevIN đầu vào.
    Khi train cộng noise Gaussian std = 1/E vào logits; chọn top-k, softmax trên k logit.
    dispatch="dense": chạy mọi expert rồi nhân mặt nạ gate (đối chứng).
    dispatch="sparse": mỗi expert chỉ chạy trên các token được route tới nó, rồi index_add_.
    Sau mỗi forward: self.aux_loss (Switch load-balance, chỉ khi train) và
    self.expert_counts (số lượt gán token → expert, [E]).
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 1,
                 dropout: float = 0.1, dispatch: str = "sparse"):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if dispatch not in ("dense", "sparse"):
            raise ValueError("dispatch must be dense or sparse")
        self.num_experts = num_experts
        self.top_k = top_k
        self.dispatch = dispatch
        self.shared = _ffn(d_model, max(1, d_ff // 2), dropout)
        self.experts = nn.ModuleList([
            _ffn(d_model, max(1, d_ff // (2 * top_k)), dropout) for _ in range(num_experts)])
        self.router = nn.Linear(d_model + 2, num_experts)
        self.aux_loss = None
        self.expert_counts = None

    def forward(self, x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """x : [B, C, D], mu/log_sigma : [B, C, 1] → [B, C, D]"""
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])                                   # [N, D]
        logits = self.router(torch.cat([x, mu, log_sigma], dim=-1)).reshape(-1, self.num_experts)
        if self.training:
            logits = logits + torch.randn_like(logits) / self.num_experts
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)              # [N, k]
        gates = F.softmax(top_logits, dim=-1)                               # [N, k]

        out = self.shared(tokens)
        if self.dispatch == "dense":
            # [N, k, E] → [N, E]: gate tại expert được chọn, 0 ở chỗ khác.
            weights = (F.one_hot(top_idx, self.num_experts).to(gates.dtype) * gates.unsqueeze(-1)).sum(1)
            expert_out = torch.stack([expert(tokens) for expert in self.experts], dim=1)   # [N, E, D]
            out = out + (weights.unsqueeze(-1) * expert_out).sum(1)
        else:
            flat_idx = top_idx.reshape(-1)                                  # [N·k]
            flat_gate = gates.reshape(-1, 1)
            token_ids = torch.arange(tokens.shape[0], device=x.device).repeat_interleave(self.top_k)
            routed = torch.zeros_like(tokens)
            for e, expert in enumerate(self.experts):
                sel = (flat_idx == e).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                ids = token_ids[sel]
                routed.index_add_(0, ids, expert(tokens[ids]) * flat_gate[sel])
            out = out + routed

        counts = torch.bincount(top_idx.reshape(-1), minlength=self.num_experts)
        self.expert_counts = counts.detach()
        if self.training:
            # Switch: aux = E · Σ f_i · P_i; f_i = tỉ lệ lượt gán top-k (chia k), P_i = xác suất router.
            frac = counts.to(logits.dtype) / (tokens.shape[0] * self.top_k)
            prob = F.softmax(logits, dim=-1).mean(0)
            self.aux_loss = self.num_experts * (frac * prob).sum()
        else:
            self.aux_loss = None
        return out.reshape(shape)
