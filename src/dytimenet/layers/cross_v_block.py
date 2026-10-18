"""Cross-variable mixing with DropEdge and DropNeuron (DropFusion)."""

import torch
from torch import nn
from torch.nn import functional as F


class DropFusionLinear(nn.Module):
    """Linear kèm DropEdge (p1, trên trọng số) và DropNeuron (p2, trên đầu ra)."""
    def __init__(self, in_features: int, out_features: int,
                 p1: float = 0.5, p2: float = 0.5, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.p1, self.p2  = p1, p2
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            m1 = torch.bernoulli(torch.full_like(self.weight, 1.0 - self.p1))
            w_masked = (self.weight * m1) / (1.0 - self.p1 + 1e-8)
            out = F.linear(x, w_masked, self.bias)
            m2 = torch.bernoulli(torch.full(out.shape, 1.0 - self.p2,
                                            device=out.device, dtype=out.dtype))
            out = (out * m2) / (1.0 - self.p2 + 1e-8)
            return out
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"p1(edge)={self.p1}, p2(neuron)={self.p2}")


class CrossVBlock(nn.Module):
    """Cross-Variable Block với DropFusion (học phụ thuộc giữa các biến).

    residual_from_norm=False: trả f(LN(x)) + x; True: trả f(LN(x)) + LN(x) (eq. 2.14 / eq. 25 paper).
    """
    def __init__(self, num_vars: int, d_model: int,
                 p1: float = 0.5, p2: float = 0.5, hidden_ratio: float = 0.75,
                 residual_from_norm: bool = False):
        super().__init__()
        self.residual_from_norm = residual_from_norm
        c_hidden = max(1, int(num_vars * hidden_ratio))
        self.norm = nn.LayerNorm(d_model)
        self.df_linear1 = DropFusionLinear(num_vars, c_hidden, p1=p1, p2=p2)
        self.df_linear2 = DropFusionLinear(c_hidden, num_vars, p1=p1, p2=p2)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, D] → [B, C, D]"""
        residual = x
        x = self.norm(x)
        if self.residual_from_norm:
            residual = x
        x = x.transpose(1, 2)      # [B, D, C]
        x = self.df_linear1(x)
        x = self.activation(x)
        x = self.df_linear2(x)
        x = x.transpose(1, 2)      # [B, C, D]
        return x + residual
