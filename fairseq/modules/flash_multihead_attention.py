# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple
from torch.nn import Parameter
from fairseq import utils

import torch
from torch import nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.multihead_attention import MultiheadAttention

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    raise ImportError("Please install the flash_attn>=2.5.6")

try:
    from rotary_embedding_torch import RotaryEmbedding
except ImportError:
    raise ImportError("Please install the rotary-embedding-torch>=0.6.4")


# HACK: This attention variant is mainly for speedup.
# HACK: Attenion weights are internalized and None is returned for them.
# HACK: Double check your requirements before using this variant.
# HACK: THis module needs extensive testing before using it in production.


class FlashMultiheadAttention(MultiheadAttention):
    """Flash Multi-headed attention
    A Flash Attention version of NativeMultiheadAttention.
    Copied from: https://huggingface.co/ai4bharat/indictrans2-en-indic-1B/blob/main/modeling_indictrans.py
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
        use_alibi=False,
        is_decoder=False,
    ):
        super().__init__(embed_dim, num_heads, dictionary=dictionary)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.is_decoder = is_decoder
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.alibi = use_alibi and self.self_attention
        self.rope = use_rope and self.self_attention

        # partial rotation in RoPE
        self.rotary_pos_embed = (
            RotaryEmbedding(
                theta=10000,
                dim=(
                    self.head_dim // 2
                ),  # partial rotation works better than full rotation
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

        self.alibi_slopes = (
            torch.Tensor(utils.get_alibi_slopes(num_heads)).float()
            if self.alibi
            else None
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()
        self.init_incremental_state()

    def _get_unpad_data(self, attn_mask):
        seqlens_in_batch = attn_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attn_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

    def _flash_attention_forward(
        self,
        q,
        k,
        v,
        attn_mask,
        dropout=0.0,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            q (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            k (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            v (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attn_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        causal = self.is_decoder and self.self_attention

        # Contains at least one padding token in the sequence
        if attn_mask is not None:
            bsz, tgt_len, _, _ = q.shape
            (
                q,
                k,
                v,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(q, k, v, attn_mask)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, bsz, tgt_len)
        else:
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
            )

        return attn_output

    def _upad_input(self, q, k, v, attn_mask):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = self._get_unpad_data(attn_mask)
        bsz, src_len, num_heads, head_dim = k.shape

        tgt_len = q.size(1)

        k = index_first_axis(
            k.reshape(bsz * src_len, num_heads, head_dim),
            indices_k,
        )
        v = index_first_axis(
            v.reshape(bsz * src_len, num_heads, head_dim),
            indices_k,
        )
        if tgt_len == src_len:
            # during training
            q = index_first_axis(
                q.reshape(bsz * src_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif tgt_len == 1:
            # during inference
            q = q.squeeze(1)
            cu_seqlens_q = torch.arange(bsz + 1, dtype=torch.int32, device=q.device)
            max_seqlen_in_batch_q = 1
            indices_q = cu_seqlens_q[:-1]
        else:
            # The -q_len: slice assumes left padding.
            q, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                q, attn_mask[:, -tgt_len:]
            )

        return (
            q,
            k,
            v,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

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

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k=k,
                v=v,
                bsz=bsz,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

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
            key_padding_mask = FlashMultiheadAttention._append_prev_key_padding_mask(
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

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                # convert to bool in case user provides a FloatTensor
                key_padding_mask = key_padding_mask.bool()
            # invert the key_padding_mask
            # as the key_padding_mask is 1 for padding tokens and 0 for non-padding tokens
            key_padding_mask = ~key_padding_mask

        # q, k, v should be (batch_size, timesteps, num_heads, head_dim)
        q = rearrange(q, "(b h) t d -> b t h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "(b h) t d -> b t h d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "(b h) t d -> b t h d", h=self.num_heads, d=self.head_dim)

        attn = self._flash_attention_forward(
            q=q, k=k, v=v, attn_mask=key_padding_mask, dropout=self.dropout_p
        )

        # the attn tensor would be of shape (batch_size, timesteps, num_heads, head_dim)
        attn = rearrange(
            attn, "b t h c -> t b (h c)", h=self.num_heads, c=self.head_dim
        )
        attn = self.out_proj(attn)
        return attn, None
