import torch
import torch.nn as nn

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


def RMSNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_rmsnorm:
        return FusedRMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
    return nn.RMSNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )
