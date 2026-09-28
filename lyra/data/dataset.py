import json
import torch

from datasets import load_dataset

from lyra.data.format import format_decision_input
from lyra.data.tokenize import LyraTokenizer


DATASET_NAME = "LocalLLaMA/typed-decisions"

WORKFLOWS = [
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
]


def load_typed_decisions(
    workflow: str,
    split: str = "train",
):
    if workflow not in WORKFLOWS:
        raise ValueError(
            f"Unknown workflow: {workflow}. "
            f"Available: {WORKFLOWS}"
        )

    return load_dataset(
        DATASET_NAME,
        workflow,
        split=split,
    )


def parse_example(row: dict) -> dict:
    return {
        "id": row["id"],
        "workflow": row["workflow"],
        "split": row["split"],
        "state": json.loads(row["state"]),
        "questions": json.loads(row["questions"]),
        "gold": json.loads(row["gold"]),
        "factors": json.loads(row["factors"]),
    }


class DecisionDataset:
    """
    PyTorch-style dataset wrapper around typed-decisions.

    Each dataset item corresponds to one decision question.
    """

    def __init__(
        self,
        workflow: str,
        split: str,
        question_name: str,
        tokenizer: LyraTokenizer | None = None,
    ):
        self.dataset = load_typed_decisions(
            workflow,
            split,
        )

        self.question_name = question_name

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else LyraTokenizer()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        row = self.dataset[index]

        example = parse_example(row)

        formatted = format_decision_input(
            example,
            self.question_name,
        )

        encoded = self.tokenizer.tokenize_decision(
            formatted
        )

        probabilities = example["gold"][
            self.question_name
        ]["probabilities"]

        target = torch.tensor(
            [
                probabilities[name]
                for name in encoded["option_names"]
            ],
            dtype=torch.float32,
        )

        encoded["target"] = target

        return encoded