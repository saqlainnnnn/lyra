import torch

from lyra.model.rope import RotaryEmbedding


def main():
    head_dim = 4

    rope = RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=10,
    )

    x = torch.zeros(
        1,
        1,
        3,
        head_dim,
    )

    # Same vector at every position.
    x[0, 0, :, 0] = 1.0

    output = rope(x)

    print("input:")
    print(x)

    print("\noutput:")
    print(output)

    print("\nposition 0:")
    print(output[0, 0, 0])

    print("\nposition 1:")
    print(output[0, 0, 1])

    print("\nposition 2:")
    print(output[0, 0, 2])


if __name__ == "__main__":
    main()