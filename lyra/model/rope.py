import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE"
            )

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        sequence_length = x.shape[-2]

        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=x.device,
            )

        frequencies = torch.outer(
            position_ids.to(self.inv_freq.dtype),
            self.inv_freq,
        )

        cos = torch.cos(frequencies)
        sin = torch.sin(frequencies)

        cos = torch.repeat_interleave(
            cos,
            2,
            dim=-1,
        )

        sin = torch.repeat_interleave(
            sin,
            2,
            dim=-1,
        )

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rotated = torch.stack(
            [
                x1 * cos[..., ::2]
                - x2 * sin[..., ::2],

                x1 * sin[..., ::2]
                + x2 * cos[..., ::2],
            ],
            dim=-1,
        )

        return rotated.flatten(-2)