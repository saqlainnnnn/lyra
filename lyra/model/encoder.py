import torch
import torch.nn as nn

from lyra.model.embeddings import TokenEmbedding
from lyra.model.block import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 8192,
        global_attn_every_n_layers: int = 3,
        local_attention: int = 128,
        local_rope_theta: float = 10000.0,
        global_rope_theta: float = 160000.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        norm_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    layer_idx=i,
                    global_attn_every_n_layers=global_attn_every_n_layers,
                    local_attention=local_attention,
                    local_rope_theta=local_rope_theta,
                    global_rope_theta=global_rope_theta,
                    attention_bias=attention_bias,
                    mlp_bias=mlp_bias,
                    norm_bias=norm_bias,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-5,
            elementwise_affine=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.embedding(input_ids)

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=input_ids.device,
            )

        for layer in self.layers:
            x = layer(
                x,
                attention_mask,
                position_ids,
            )

        x = self.final_norm(x)

        return x