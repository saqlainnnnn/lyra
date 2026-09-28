import torch

from lyra.model.decision import DecisionHead


def main():
    batch_size = 2
    num_options = 5
    hidden_size = 192

    candidate_hidden = torch.randn(
        batch_size,
        num_options,
        hidden_size,
    )

    head = DecisionHead(
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=3,
        intermediate_size=768,
    )

    scores = head(candidate_hidden)

    print("candidate hidden:")
    print(candidate_hidden.shape)

    print("\nscores:")
    print(scores.shape)

    print("\nparameters:")
    print(sum(p.numel() for p in head.parameters()))

    print("\nscores:")
    print(scores)

    print("\nrequires gradient:")
    print(scores.requires_grad)


if __name__ == "__main__":
    main()