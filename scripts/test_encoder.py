import torch

from lyra.model.encoder import TransformerEncoder


def main():

    batch_size = 2
    sequence_length = 10

    vocab_size = 50368
    hidden_size = 256
    num_layers = 8
    num_heads = 4
    intermediate_size = 768

    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long,
    )

    encoder = TransformerEncoder(
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
        dropout=0.0,
    )

    output = encoder(
        input_ids,
        attention_mask,
    )

    print("input ids:")
    print(input_ids.shape)

    print("\noutput:")
    print(output.shape)

    print("\nparameters:")
    print(sum(p.numel() for p in encoder.parameters()))

    print("\noutput requires gradient:")
    print(output.requires_grad)


if __name__ == "__main__":
    main()