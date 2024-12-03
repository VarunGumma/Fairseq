import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    def __init__(
        self,
        r: int,
        alpha: int,
        dropout: float = 0.0,
        rank_scaled: bool = False,
        fan_in: int = None,
        fan_out: int = None,
    ):
        assert r > 0, "Rank must be greater than 0 for LoRA to work."

        self.merged = False
        self.dropout_p = dropout
        self.scaling = (alpha / math.sqrt(r)) if rank_scaled else (alpha / r)
        # better to contain the A and B matrices in the same class
        self.lora_A = nn.Parameter(torch.zeros((r, fan_in)))
        self.lora_B = nn.Parameter(torch.zeros((fan_out, r)))


class LoRAEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        alpha: int = 1,
        dropout: float = 0.0,
        rank_scaled: bool = False,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            ranked_scaled=rank_scaled,
            fan_in=embedding_dim,
            fan_out=num_embeddings,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
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
            x = F.embedding(
                x,
                self.lora_A.T,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            result += (x @ self.lora_B.T) * self.scaling

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
        **kwargs,
    ):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            rank_scaled=rank_scaled,
            fan_in=in_features,
            fan_out=out_features,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, r={self.lora_A.shape[0]})"

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
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
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            result += x @ ((self.lora_B @ self.lora_A).T * self.scaling)

        return result
