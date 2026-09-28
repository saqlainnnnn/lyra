import torch

from lyra.model.block import TransformerBlock


def main():

    batch_size = 2
    sequence_length = 10

    hidden_size = 256
    num_heads = 4
    intermediate_size = 768

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

    block = TransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        layer_idx=0,
        global_attn_every_n_layers=3,
        local_attention=128,
        local_rope_theta=10000.0,
        global_rope_theta=160000.0,
        attention_bias=False,
        mlp_bias=False,
        norm_bias=False,
        dropout=0.0,
    )

    output = block(
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
        for p in block.parameters()
    ))

    print("\noutput requires gradient:")
    print(output.requires_grad)

    loss = output.sum()
    loss.backward()

    print("\nbackward:")
    print("success")


if __name__ == "__main__":
    main()