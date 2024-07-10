import torch
from torch import nn

try:
    from apex.normalization import FusedRMSNorm as _FusedRMSNorm

    has_fused_rmsnorm = True

    class FusedRMSNorm(_FusedRMSNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_rmsnorm = False


class _RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


def RMSNorm(
    normalized_shape,
    eps=1e-5,
    elementwise_affine=True,
    memory_efficient=False,
    export=False,
):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_rmsnorm:
        return FusedRMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            memory_efficient=memory_efficient,
        )
    return _RMSNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )
