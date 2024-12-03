import torch
from torch.amp import autocast
from torch import einsum
from einops import rearrange, repeat

device = "cuda" if torch.cuda.is_available() else "cpu"


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(cos, sin, t):
    rot_dim = cos.shape[-1]
    assert rot_dim <= t.shape[-1] and cos.shape == sin.shape
    t_left, t_right = t[..., :rot_dim], t[..., rot_dim:]
    t_transformed = (t_left * cos) + (rotate_half(t_left) * sin)
    return torch.cat((t_transformed, t_right), dim=-1).type(t.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, dim, theta=10000, interpolate_factor=1.0, cache_max_seq_len=8192
    ):
        super().__init__()
        self.theta = theta

        freqs_ = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.cache_max_seq_len = cache_max_seq_len
        self.interpolate_factor = interpolate_factor

        self.freqs = torch.nn.Parameter(freqs_, requires_grad=False).to(device)
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)
        self.precompute_freqs(cache_max_seq_len)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim={self.freqs.shape[-1]}, theta={self.theta})"
        )

    def precompute_freqs(self, max_seq_len):
        thetas = self.forward(max_seq_len, device=device)
        self.register_buffer("cached_cos", thetas.cos(), persistent=False)
        self.register_buffer("cached_sin", thetas.sin(), persistent=False)

    def rotate_queries_or_keys(self, t, seq_dim=-2, offset=0):
        seq_len = t.shape[seq_dim]

        if seq_len > self.cache_max_seq_len:
            self.cache_max_seq_len = seq_len * 2
            self.precompute_freqs(self.cache_max_seq_len)

        cos, sin = (
            self.cached_cos[offset : (offset + seq_len)],
            self.cached_sin[offset : (offset + seq_len)],
        )
        return apply_rotary_emb(cos, sin, t)

    @autocast("cuda", enabled=False)
    def forward(self, seq_len, device):
        seq = torch.arange(seq_len, device=device) / self.interpolate_factor
        thetas = einsum("..., f -> ... f", seq, self.freqs)
        thetas = repeat(thetas, "... n -> ... (n r)", r=2)
        return thetas
