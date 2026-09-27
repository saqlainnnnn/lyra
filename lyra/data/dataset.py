import json

from datasets import load_dataset


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
    """Load one typed-decisions workflow."""

    if workflow not in WORKFLOWS:
        raise ValueError(
            f"Unknown workflow: {workflow}. "
            f"Available workflows: {WORKFLOWS}"
        )

    return load_dataset(
        DATASET_NAME,
        workflow,
        split=split,
    )


def parse_example(row: dict) -> dict:
    """Convert JSON-encoded dataset fields into Python objects."""

    return {
        "id": row["id"],
        "workflow": row["workflow"],
        "split": row["split"],
        "state": json.loads(row["state"]),
        "questions": json.loads(row["questions"]),
        "gold": json.loads(row["gold"]),
        "factors": json.loads(row["factors"]),
    }

def build_decision_input(example: dict, question_name: str) -> dict:
    """Build the model input for one decision question."""

    if question_name not in example["questions"]:
        raise ValueError(
            f"Unknown question: {question_name}. "
            f"Available: {list(example['questions'])}"
        )

    question = example["questions"][question_name]

    return {
        "question_name": question_name,
        "state": example["state"],
        "instruction": question["instructions"],
        "criteria": question["criteria"],
        "type": question["type"],
    }