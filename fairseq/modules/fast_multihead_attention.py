# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple
from torch.nn import Parameter

import json
import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from fairseq.modules.quant_noise import quant_noise
from torch.nn.functional import scaled_dot_product_attention
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.rotary_embedding import RotaryEmbedding

# HACK: This attention variant is mainly for speedup.
# HACK: Attenion weights are internalized and None is returned for them.
# HACK: Double check your requirements before using this variant.


class FastMultiheadAttention(MultiheadAttention):
    """Fast Multi-headed attention
    A Faster version of NativeMultiheadAttention that uses a kernelized implementation of scaled dot product attention.
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
        is_decoder=False,
        rope_args=None,
        fused_qkv=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            dictionary=dictionary,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        del self.dropout_module

        self.is_decoder = is_decoder
        self.fused_qkv = fused_qkv
        self.rope = rope_args is not None and self_attention

        if self.rope:
            rope_args = json.loads(rope_args)
            
        # partial rotation in RoPE
        self.rotary_pos_embed = (
            RotaryEmbedding(
                dim=self.head_dim // 2,
                theta=rope_args.get("theta", 10000),
                scaling_factor=rope_args.get("scaling_factor", 1.0),
            )
            if self.rope
            else None
        )

        if self.fused_qkv:
            if self_attention:
                # fused qkv for self-attention
                # all of W_q, W_k, W_v are fused into one weight
                del self.q_proj, self.k_proj, self.v_proj
                self.qkv_proj = quant_noise(
                    nn.Linear(embed_dim, 3 * embed_dim, bias=bias),
                    q_noise,
                    qn_block_size,
                )
            elif encoder_decoder_attention:
                # in case of encoder-decoder attention, q is from encoder and kv is from decoder
                # only W_k and W_v are fused
                del self.k_proj, self.v_proj
                self.kv_proj = quant_noise(
                    nn.Linear(embed_dim, 2 * embed_dim, bias=bias),
                    q_noise,
                    qn_block_size,
                )
            else:
                raise NotImplementedError(
                    "Fused qkv is only supported for self-attention or encoder-decoder attention. Either of the two must be True."
                )

        self.reset_parameters()

    def _apply_rotary_pos_emb(self, q, k, is_inference=False):
        offset = (k.shape[-2] - 1) if is_inference else 0
        q = self.rotary_pos_embed.rotate_queries_or_keys(q, offset=offset)
        k = self.rotary_pos_embed.rotate_queries_or_keys(k)
        return q, k

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            if hasattr(self, "qkv_proj"):
                nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(2))
            elif hasattr(self, "kv_proj"):
                nn.init.xavier_uniform_(self.kv_proj.weight, gain=1 / math.sqrt(2))
            else:
                nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

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

        dropout_p = self.dropout_p if self.training else 0

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
            if not self.fused_qkv:
                q = self.q_proj(query)
                k = self.k_proj(query)
                v = self.v_proj(query)
            else:
                q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if not self.fused_qkv:
                    k = self.k_proj(key)
                    v = self.v_proj(key)
                else:
                    k, v = self.kv_proj(key).chunk(2, dim=-1)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k=k,
                v=v,
                bsz=bsz,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        q = rearrange(q, "t b (h d) -> (b h) t d", h=self.num_heads, d=self.head_dim)
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = rearrange(
                k,
                "t b (h d) -> (b h) t d",
                h=self.num_heads,
                d=self.head_dim,
            )
        if v is not None:
            v = rearrange(
                v,
                "t b (h d) -> (b h) t d",
                h=self.num_heads,
                d=self.head_dim,
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
            q, k = self._apply_rotary_pos_emb(
                q, k, is_inference=saved_state is not None
            )

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        # create one single mask for causality and padding
        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.num_heads, tgt_len, -1)
                .reshape(bsz * self.num_heads, tgt_len, src_len)
                .to(torch.bool)
                .float()
                * torch.finfo(q.dtype).min
            )

            # SDPA cannot accept both causal and attn_mask
            # So, we combine them here
            combined_mask = (
                (
                    (attn_mask.unsqueeze(0) + key_padding_mask)
                    if attn_mask.size() != key_padding_mask.size()
                    else attn_mask + key_padding_mask
                )
                if attn_mask is not None
                else key_padding_mask
            )

        else:
            combined_mask = attn_mask

        combined_mask = combined_mask.to(q.dtype) if combined_mask is not None else None

        attn = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False,
            attn_mask=combined_mask,
            dropout_p=dropout_p,
        )

        attn = rearrange(
            attn, "(b h) t d -> t b (h d)", h=self.num_heads, d=self.head_dim
        )
        attn = self.out_proj(attn)
        return attn, None
