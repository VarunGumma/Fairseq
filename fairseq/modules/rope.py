# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch


class ROPE(torch.nn.Module):
    def __init__(self, head_dim, base=10000, max_positions=512):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        sin_, cos_ = self.fixed_pos_embedding(max_positions)
        self.register_buffer("sin", sin_)
        self.register_buffer("cos", cos_)

    def fixed_pos_embedding(self, seq_len):
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.int64).float()
                / self.head_dim
            )
        )
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
        )
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    def duplicate_interleave(self, m):
        dim0 = m.shape[0]
        return m.view(-1, 1).repeat(1, 2).view(dim0, -1)

    def apply_rotary_pos_emb(self, x, sin, cos):
        sin, cos = map(lambda t: self.duplicate_interleave(t), (sin, cos))
        return (x * cos) + (self.rotate_every_two(x) * sin)

    @torch.no_grad()
    def forward(self, x):
        """
        x: [bsz * n_heads, seq_len, dim]
        """
        seq_len = x.shape[1]

        if seq_len > self.sin.shape[0]:
            sin_, cos_ = self.fixed_pos_embedding(seq_len)
            self.sin.data = sin_
            self.cos.data = cos_
        else:
            sin_, cos_ = self.sin[:seq_len], self.cos[:seq_len]

        return self.apply_rotary_pos_emb(x, sin=sin_, cos=cos_)
