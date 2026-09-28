import math

import torch
import torch.nn as nn

from lyra.model.rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        local_attention: int = 128,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_global_attention: bool = False,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_attention = local_attention
        self.use_global_attention = use_global_attention

        self.qkv = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=attention_bias,
        )

        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=attention_bias,
        )

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=8192,
            base=rope_theta,
        )

    def _local_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        positions = torch.arange(
            sequence_length,
            device=device,
        )

        distance = (
            positions[:, None] - positions[None, :]
        ).abs()

        return distance < (self.local_attention // 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        qkv = self.qkv(x)

        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :]

            scores = scores.masked_fill(
                padding_mask == 0,
                torch.finfo(scores.dtype).min,
            )

        if not self.use_global_attention:
            local_mask = self._local_mask(
                sequence_length,
                x.device,
            )

            scores = scores.masked_fill(
                ~local_mask[None, None, :, :],
                torch.finfo(scores.dtype).min,
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        context = torch.matmul(
            attention_weights,
            v,
        )

        context = context.transpose(
            1,
            2,
        ).contiguous()

        context = context.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        return self.out_proj(context)