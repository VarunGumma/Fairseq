import torch.nn as nn


class FactorizedEmbedding(nn.Module):
    """
    Factorized Embedding from "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (Lan et al.,)
    <https://arxiv.org/abs/1909.11942>

    Args:
        num_embeddings: vocabulary size
        embedding_dim: Final embedding dimension
        hid_dim: factored lower dimension for embedding vectors
        padding_idx: pad token index in the vocabulary
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        hid_dim=128,
        padding_idx=1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim

        self.up = nn.Linear(hid_dim, embedding_dim, bias=False)
        self.m = nn.Embedding(num_embeddings, hid_dim, padding_idx=padding_idx)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.m.weight, mean=0, std=self.embedding_dim**-0.5)
        nn.init.constant_(self.m.weight[self.padding_idx], 0)

    def forward(self, x):
        return self.up(self.m(x))
