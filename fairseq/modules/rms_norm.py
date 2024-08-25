import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"
