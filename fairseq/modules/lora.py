import torch
import math
import torch.nn as nn
import torch.nn.functional as F


# source: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class LoRALayer:
    def __init__(
        self,
        r: int,
        alpha: int,
        dropout: float,
        rank_scaled: bool,
    ):

        assert (
            r > 0
        ), "rank must be greater than 0 for LoRA to work. Use nn.Linear/nn.Embedding if rank is 0."

        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        # Mark the weight as unmerged initially
        self.merged = False
        self.rank_scaled = rank_scaled


class LoRAEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        alpha: int = 1,
        rank_scaled: bool = False,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, alpha=alpha, dropout=0, rank_scaled=rank_scaled)
        # Actual trainable parameters
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
        self.scaling = (
            (self.alpha / math.sqrt(self.r)) if rank_scaled else (self.alpha / self.r)
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merged:
                # Make sure that the weights are not merged in training mode
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
                self.merged = False
        else:
            if not self.merged:
                # During evaluation, if the weights are not merged, merge them
                self.weight.data += (self.lora_B @ self.lora_A).T * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        result = nn.Embedding.forward(self, x)

        if not self.merged:
            after_A = F.embedding(
                x,
                self.lora_A.T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.lora_B.T) * self.scaling

        return result


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: int = 1,
        dropout: float = 0.0,
        rank_scaled: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=r, alpha=alpha, dropout=dropout, rank_scaled=rank_scaled
        )
        # Actual trainable parameters
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = (
            (self.alpha / math.sqrt(self.r)) if rank_scaled else (self.alpha / self.r)
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but should not affect performance
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merged:
                # Make sure that the weights are not merged in training mode
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if not self.merged:
                # During evaluation, if the weights are not merged, merge them
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)

        if not self.merged:
            # if the weights are not merged, apply LoRA
            if self.dropout is not None:
                x = self.dropout(x)
            result += x @ ((self.lora_B @ self.lora_A).T * self.scaling)

        return result
