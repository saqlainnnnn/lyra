import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = 50283,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_token_id,
        )

        self.norm = nn.LayerNorm(
            hidden_size,
            eps=1e-5,
            elementwise_affine=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.norm(x)
        x = self.dropout(x)
        return x