"""Reversible instance normalization along the time axis."""

from typing import Optional
import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (chuẩn hóa theo chiều thời gian)."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))
        self._mean: Optional[torch.Tensor] = None
        self._std:  Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=-1, keepdim=True).detach()
            self._std  = x.std(dim=-1, keepdim=True, unbiased=False).detach() + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight[None, :, None] + self.affine_bias[None, :, None]
            return x
        elif mode == "denorm":
            if self._mean is None:
                raise RuntimeError("Gọi denorm trước khi gọi norm!")
            if self.affine:
                x = (x - self.affine_bias[None, :, None]) / (self.affine_weight[None, :, None] + self.eps)
            return x * self._std + self._mean
        else:
            raise ValueError(f"mode phải là 'norm' hoặc 'denorm', nhận: '{mode}'")
