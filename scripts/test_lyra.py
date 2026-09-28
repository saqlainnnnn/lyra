import torch

from lyra.model.lyra import Lyra


def main():

    model = Lyra(
        vocab_size=50368,
        hidden_size=256,
        num_layers=8,
        num_heads=4,
        intermediate_size=768,
        decision_num_layers=2,
        decision_num_heads=4,
        decision_intermediate_size=768,
        dropout=0.0,
    )

    input_ids = torch.randint(
        0,
        50368,
        (2, 20),
    )

    attention_mask = torch.ones(
        2,
        20,
        dtype=torch.long,
    )

    marker_positions = torch.tensor(
        [
            [2, 7, 12, 15, 18],
            [1, 5, 9, 13, 17],
        ],
        dtype=torch.long,
    )

    scores = model(
        input_ids,
        attention_mask,
        marker_positions,
    )

    print("scores:")
    print(scores)

    print("\nshape:")
    print(scores.shape)

    print("\nparameters:")
    print(sum(
        p.numel()
        for p in model.parameters()
    ))

    print("\nrequires gradient:")
    print(scores.requires_grad)

    loss = scores.sum()
    loss.backward()

    print("\nbackward:")
    print("success")


if __name__ == "__main__":
    main()