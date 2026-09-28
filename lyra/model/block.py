import torch
import torch.nn as nn

from lyra.model.attention import MultiHeadSelfAttention


class ModernBERTMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.wi = nn.Linear(
            hidden_size,
            intermediate_size * 2,
            bias=bias,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.wo = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.wi(x).chunk(2, dim=-1)

        x = self.activation(x) * gate

        x = self.dropout(x)

        return self.wo(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_idx: int,
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

        self.layer_idx = layer_idx

        self.use_global_attention = (
            layer_idx % global_attn_every_n_layers == 0
        )

        self.attention_norm = (
            nn.Identity()
            if layer_idx == 0
            else nn.LayerNorm(
                hidden_size,
                eps=1e-5,
                elementwise_affine=True,
            )
        )

        rope_theta = (
            global_rope_theta
            if self.use_global_attention
            else local_rope_theta
        )

        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            local_attention=local_attention,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            use_global_attention=self.use_global_attention,
        )

        self.mlp_norm = nn.LayerNorm(
            hidden_size,
            eps=1e-5,
            elementwise_affine=True,
        )

        self.mlp = ModernBERTMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        residual = x

        x = self.attention_norm(x)

        x = self.attention(
            x,
            attention_mask,
            position_ids,
        )

        x = residual + x

        residual = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = residual + x

        return x