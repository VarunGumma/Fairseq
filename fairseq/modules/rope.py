# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn


class ROPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        self.base = base
        self.head_dim = head_dim

    def fixed_pos_embedding(self, x):
        seq_len, dim = x.shape
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
        ).to(x)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)

    def duplicate_interleave(self, m):
        dim0 = m.shape[0]
        return m.view(-1, 1).repeat(1, 2).view(dim0, -1)

    def apply_rotary_pos_emb(self, x, sin, cos, scale=1.0):
        sin, cos = map(lambda t: self.duplicate_interleave(t * scale), (sin, cos))
        return (x * cos) + (self.rotate_every_two(x) * sin)

    def forward(self, x):
        """
        x: [bsz * n_heads, seq_len, dim]
        """
        sin, cos = self.fixed_pos_embedding(x)
        return self.apply_rotary_pos_emb(x, sin, cos)
