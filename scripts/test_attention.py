import torch

from lyra.model.attention import MultiHeadSelfAttention


def main():

    batch_size = 2
    sequence_length = 10

    hidden_size = 256
    num_heads = 4

    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )

    position_ids = torch.arange(
        sequence_length,
    )

    attention = MultiHeadSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        local_attention=128,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_global_attention=True,
    )

    output = attention(
        x,
        attention_mask,
        position_ids,
    )

    print("input:")
    print(x.shape)

    print("\noutput:")
    print(output.shape)

    print("\nparameters:")
    print(sum(
        p.numel()
        for p in attention.parameters()
    ))

    print("\noutput requires gradient:")
    print(output.requires_grad)


if __name__ == "__main__":
    main()