"""Learnable recurrent cycles của RCF (CycleNet, Lin et al., NeurIPS 2024).

Q ∈ R^{W×C} (khởi tạo 0) là một chu kỳ dùng chung cho mọi mẫu của từng kênh. Đoạn chu kỳ ứng với
một cửa sổ bắt đầu ở pha φ = (chỉ số dòng) mod W là Q[(φ + k) mod W], k = 0..length−1.
"""

import torch
from torch import nn


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len: int, num_vars: int):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, num_vars))

    def forward(self, index: torch.Tensor, length: int) -> torch.Tensor:
        """index: [B] pha bắt đầu → đoạn chu kỳ [B, C, length]."""
        steps = torch.arange(length, device=index.device)
        gather = (index.view(-1, 1) + steps.view(1, -1)) % self.cycle_len      # [B, length]
        return self.data[gather].transpose(1, 2)
