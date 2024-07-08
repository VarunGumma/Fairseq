# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.multihead_attention import MultiheadAttention

# HACK: This attention variant is mainly for speedup, and the attenion weights are internalized and None is returned for them.
# Arguments like `add_bias_kv` and `add_zero_attn` are not used in this implementation. So, double check your requirements before using this variant.


class FastMultiheadAttention(MultiheadAttention):
    """Native Multi-headed attention
    Removes a lot of the overhead in the MultiheadAttention module
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
        use_rope=False,
    ):
        super().__init__(embed_dim, num_heads, dictionary=dictionary)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.dropout_p = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.rope = use_rope and self.self_attention

        # partial rotation in RoPE
        self.rotary_pos_embed = (
            RotaryEmbedding(
                theta=10000,
                dim=(self.head_dim // 2),
                freqs_for="lang",
                use_xpos=False,
                seq_before_head_dim=False,
                cache_if_possible=True,
                learned_freq=False,
            )
            if self.rope
            else None
        )

        assert (
            not self.self_attention or self.qkv_same_dim
        ), "Self-attention requires query, key and value to be of the same size"

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.beam_size = 1
        self.reset_parameters()

        self.init_incremental_state()

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        need_weights: bool = False,
        static_kv: bool = False,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = FastMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                kv_bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        assert k.size(1) == src_len

        if self.rotary_pos_embed is not None:
            q = rearrange(q, "(b h) t d -> b h t d", h=self.num_heads)
            k = rearrange(k, "(b h) t d -> b h t d", h=self.num_heads)

            if saved_state is not None:
                # inference with kv caching
                q, k = self.rotary_pos_embed.rotate_queries_with_cached_keys(q, k)
            else:
                q = self.rotary_pos_embed.rotate_queries_or_keys(q)
                k = self.rotary_pos_embed.rotate_queries_or_keys(k)

            q = rearrange(q, "b h t d -> (b h) t d", h=self.num_heads)
            k = rearrange(k, "b h t d -> (b h) t d", h=self.num_heads)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.num_heads, tgt_len, -1)
                .reshape(bsz * self.num_heads, tgt_len, src_len)
                .to(torch.bool)
                .float()
                * torch.finfo(q.dtype).min
            )

            combined_mask = (
                (attn_mask.unsqueeze(0) + key_padding_mask)
                if attn_mask is not None
                else key_padding_mask
            )
        else:
            combined_mask = attn_mask

        combined_mask = combined_mask.to(q.dtype) if combined_mask is not None else None

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_cudnn=True,
            enable_mem_efficient=True,
        ):
            attn = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=combined_mask,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

        assert list(attn.size()) == [
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ], f"attn size should be {[bsz * self.num_heads, tgt_len, self.head_dim]}, but is {attn.shape()}"

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        return attn, None
