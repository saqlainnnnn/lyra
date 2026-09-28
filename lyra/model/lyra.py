import torch
import torch.nn as nn

from lyra.model.encoder import TransformerEncoder
from lyra.model.decision import DecisionHead
from lyra.model.init import initialize_modernbert


class Lyra(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        decision_num_layers,
        decision_num_heads,
        decision_intermediate_size,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=8192,
            global_attn_every_n_layers=3,
            local_attention=128,
            local_rope_theta=10000.0,
            global_rope_theta=160000.0,
            attention_bias=False,
            mlp_bias=False,
            norm_bias=False,
            dropout=dropout,
        )

        self.decision_head = DecisionHead(
            hidden_size=hidden_size,
            num_layers=decision_num_layers,
            num_heads=decision_num_heads,
            intermediate_size=decision_intermediate_size,
            dropout=dropout,
        )

        initialize_modernbert(
            self,
            initializer_range=0.02,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_positions,
        position_ids=None,
    ):
        hidden_states = self.encoder(
            input_ids,
            attention_mask,
            position_ids,
        )

        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device,
        ).unsqueeze(1)

        candidate_hidden = hidden_states[
            batch_indices,
            marker_positions,
        ]

        scores = self.decision_head(
            candidate_hidden
        )

        return scores