import argparse

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from lyra.data.collate import collate_decisions
from lyra.data.dataset import DecisionDataset
from lyra.data.tokenize import LyraTokenizer
from lyra.model.lyra import Lyra


def build_model(config):
    model_config = config["model"]
    decision_config = config["decision_head"]

    return Lyra(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        intermediate_size=model_config["intermediate_size"],
        decision_num_layers=decision_config["num_layers"],
        decision_num_heads=decision_config["num_heads"],
        decision_intermediate_size=decision_config["intermediate_size"],
        dropout=model_config["dropout"],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
    )

    parser.add_argument(
        "--workflow",
        type=str,
        default="customer_service",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="action",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = LyraTokenizer()

    dataset = DecisionDataset(
        workflow=args.workflow,
        split="train",
        question_name=args.question,
        tokenizer=tokenizer,
    )

    # Same deterministic split used by train.py.
    val_size = max(
        1,
        int(len(dataset) * 0.1),
    )

    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_decisions,
    )

    model = build_model(config)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: {args.checkpoint}"
    )

    print(
        f"Validation examples: {len(val_dataset)}"
    )

    print()

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= args.num_examples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_positions = batch["marker_positions"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                marker_positions=marker_positions,
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            candidate_mask = batch["candidate_mask"][0]
            targets = batch["targets"][0]

            valid_count = int(
                candidate_mask.sum().item()
            )

            probabilities = probabilities[0][:valid_count]
            targets = targets[:valid_count]

            option_names = batch["option_names"][0]

            print(
                "=" * 70
            )

            print(
                f"Example {index + 1}"
            )

            print()

            for i, option_name in enumerate(option_names):
                print(
                    f"{option_name}"
                )

                print(
                    f"  target:      {targets[i].item():.4f}"
                )

                print(
                    f"  prediction:  "
                    f"{probabilities[i].item():.4f}"
                )

                print(
                    f"  difference:  "
                    f"{probabilities[i].item() - targets[i].item():+.4f}"
                )

            print()

            print(
                f"Target sum:      {targets.sum().item():.6f}"
            )

            print(
                f"Prediction sum:  "
                f"{probabilities.sum().item():.6f}"
            )

            target_choice = torch.argmax(targets).item()
            predicted_choice = torch.argmax(probabilities).item()

            print(
                f"Target top option:     "
                f"{option_names[target_choice]}"
            )

            print(
                f"Predicted top option:  "
                f"{option_names[predicted_choice]}"
            )

            print()


if __name__ == "__main__":
    main()