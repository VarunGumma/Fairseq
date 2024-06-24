# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
from fairseq.modules.rope import ROPE


class XPOS(ROPE):
    def __init__(self, head_dim, base=10000, scale_base=512):
        super().__init__(base=base, head_dim=head_dim)
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def fixed_pos_embedding(self, x):
        seq_len, dim = x.shape
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim) / dim))
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
        ).to(x)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def forward(self, x, offset=0, downscale=False):
        """
        x: [bsz * n_heads, seq_len, dim]
        """
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin, cos = self.fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        return self.apply_rotary_pos_emb(x, sin, cos, scale)
