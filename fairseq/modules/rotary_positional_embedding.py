import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len: int = 0):
        """
        Args:
            x: Input x with shape (b * n_h, s, d)
            seq_len: Sequence length of input x
        """
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().view(1, emb.size(0), emb.size(1))
            self.sin_cached = emb.sin().view(1, emb.size(0), emb.size(1))
        return self.cos_cached[:, : x.shape[1]], self.sin_cached[:, : x.shape[1]]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[:, offset : q.shape[1] + offset, :],
        sin[:, offset : q.shape[1] + offset, :],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
