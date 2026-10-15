"""Moving-average trend and seasonal decomposition."""

import torch
from torch import nn


class MovingAvg(nn.Module):
    """
    Tách chuỗi thời gian thành 2 phần bằng Moving Average:
      - x_trend    : thành phần xu hướng (smoothed)
      - x_seasonal : thành phần tuần hoàn = x - x_trend

    Padding đối xứng (front/end) để giữ nguyên độ dài L.
    kernel_size nên là số lẻ để padding cân bằng hai phía.
    """
    def __init__(self, kernel_size: int = 25, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        """x : [B, C, L] → (x_seasonal, x_trend) mỗi cái [B, C, L]"""
        pad = (self.kernel_size - 1) // 2
        # Lặp giá trị đầu/cuối để padding — tránh edge artifact
        front = x[:, :, 0:1].repeat(1, 1, pad)
        end   = x[:, :, -1:].repeat(1, 1, pad)
        x_pad = torch.cat([front, x, end], dim=-1)   # [B, C, L + 2*pad]
        x_trend    = self.avg(x_pad)                  # [B, C, L]
        x_seasonal = x - x_trend
        return x_seasonal, x_trend
