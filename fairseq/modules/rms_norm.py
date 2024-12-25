import torch
import torch.nn as nn

has_fused_rmsnorm = False
has_pytorch_rmsnorm = False

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
    try:
        from torch.nn import RMSNorm as PyTorchRMSNorm

        has_pytorch_rmsnorm = True
    except ImportError:
        pass


class NativeRMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.weight

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}"


def RMSNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_rmsnorm:
        return FusedRMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
    elif has_pytorch_rmsnorm:
        return PyTorchRMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
    else:
        return NativeRMSNorm(normalized_shape, eps=eps)
