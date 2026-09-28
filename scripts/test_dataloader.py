import torch
from torch.utils.data import DataLoader

from lyra.data.dataset import DecisionDataset
from lyra.data.collate import collate_decisions
from lyra.data.tokenize import LyraTokenizer


def main():

    tokenizer = LyraTokenizer()

    dataset = DecisionDataset(
        workflow="customer_service",
        split="train",
        question_name="action",
        tokenizer=tokenizer,
    )

    print("dataset size:")
    print(len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_decisions,
    )

    batch = next(iter(loader))

    print("\ninput_ids:")
    print(batch["input_ids"].shape)

    print("\nattention_mask:")
    print(batch["attention_mask"].shape)

    print("\nmarker_positions:")
    print(batch["marker_positions"].shape)

    print("\ncandidate_mask:")
    print(batch["candidate_mask"].shape)

    print("\ntargets:")
    print(batch["targets"].shape)

    print("\noption names:")
    print(batch["option_names"])

    print("\nquestion types:")
    print(batch["question_types"])

    print("\nvalid candidates per example:")
    print(
        batch["candidate_mask"].sum(dim=1)
    )

    print("\ntarget sums:")
    print(
        batch["targets"].sum(dim=1)
    )


if __name__ == "__main__":
    main()