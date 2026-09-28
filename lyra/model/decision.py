import torch
import torch.nn as nn

from lyra.model.block import TransformerBlock


class DecisionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                layer_idx=i,
                global_attn_every_n_layers=1,
                local_attention=128,
                local_rope_theta=10000.0,
                global_rope_theta=10000.0,
                attention_bias=False,
                mlp_bias=False,
                norm_bias=False,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.output = nn.Linear(
            hidden_size,
            1,
            bias=False,
        )

    def forward(self, candidate_hidden):
        x = candidate_hidden

        for layer in self.layers:
            x = layer(x)

        scores = self.output(x).squeeze(-1)

        return scores