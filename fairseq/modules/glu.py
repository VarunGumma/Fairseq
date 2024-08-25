import torch.nn as nn

# Gated Linear Unit as proposed by Noam Shazeer in `GLU Variants Improve Transformer`
# COPIED FROM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py


# intermediate_features: --encoder-ffn-embed-dim, --decoder-ffn-embed-dim
# activation_fn: --activation-fn
# usually, bias is set to False for GLU
class GLU(nn.Module):
    def __init__(self, in_features, intermediate_features, activation_fn, bias=False):
        super().__init__()
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.activation_fn = activation_fn
        # this is a fused implementation of the two linear layers
        self.up_gate_proj = nn.Linear(
            self.in_features, 2 * self.intermediate_features, bias=bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_features, self.in_features, bias=bias
        )

    def forward(self, x):
        x = self.up_gate_proj(x)
        x, gate = x.chunk(2, dim=-1)
        return self.down_proj(x * self.activation_fn(gate))
