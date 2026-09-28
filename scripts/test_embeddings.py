import torch

from lyra.model.embeddings import TokenEmbedding


def main():
    batch_size = 2
    sequence_length = 10
    vocab_size = 50368
    hidden_size = 192

    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
    )

    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    output = embedding(input_ids)

    print("input_ids shape:")
    print(input_ids.shape)

    print("\noutput shape:")
    print(output.shape)

    print("\nembedding parameters:")
    print(sum(p.numel() for p in embedding.parameters()))


if __name__ == "__main__":
    main()