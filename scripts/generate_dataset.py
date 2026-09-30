from __future__ import annotations

import argparse
import json
from pathlib import Path

from lyra.data.generator import WORKFLOWS, generate_examples
from lyra.data.teacher import (
    GroqTeacher,
    get_option_names,
    label_example,
)


def save_jsonl(examples, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def audit_examples(examples) -> None:
    if not examples:
        raise ValueError("No examples were generated.")

    seen_ids = set()

    for example in examples:
        # ------------------------------------------------------------
        # Basic example checks
        # ------------------------------------------------------------
        example_id = example["id"]

        if example_id in seen_ids:
            raise ValueError(f"Duplicate example id: {example_id}")

        seen_ids.add(example_id)

        workflow = example["workflow"]

        if workflow not in WORKFLOWS:
            raise ValueError(f"Unknown workflow: {workflow}")

        questions = example["questions"]
        gold = example.get("gold")

        if len(questions) != 5:
            raise ValueError(
                f"Expected 5 questions, got {len(questions)} "
                f"for example {example_id}"
            )

        if gold is None:
            raise ValueError(
                f"Missing gold labels for example {example_id}"
            )

        # ------------------------------------------------------------
        # Check every question's gold distribution
        # ------------------------------------------------------------
        for question_name, question in questions.items():
            if question_name not in gold:
                raise ValueError(
                    f"Missing gold for question '{question_name}' "
                    f"in example {example_id}"
                )

            probabilities = gold[question_name]["probabilities"]

            expected_options = set(get_option_names(question))
            actual_options = set(probabilities.keys())

            if expected_options != actual_options:
                raise ValueError(
                    f"Incorrect gold options for "
                    f"{example_id}/{question_name}. "
                    f"Expected: {sorted(expected_options)}, "
                    f"Got: {sorted(actual_options)}"
                )

            total = sum(probabilities.values())

            if abs(total - 1.0) > 1e-5:
                raise ValueError(
                    f"Probabilities do not sum to 1 for "
                    f"{example_id}/{question_name}: {total}"
                )

            for option, probability in probabilities.items():
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"Invalid probability {probability} for "
                        f"{example_id}/{question_name}/{option}"
                    )

    print(f"Audit passed: {len(examples)} examples")


def generate_split(
    *,
    num_examples: int,
    split: str,
    workflow: str,
    seed: int,
    output_dir: Path,
    teacher_model: str | None = None,
    num_teacher_samples: int = 3,
    temperature: float = 0.7,
) -> None:

    print()
    print("=" * 70)
    print(f"Generating {split} split")
    print(f"Workflow: {workflow}")
    print(f"Examples: {num_examples}")
    print(f"Teacher samples/example: {num_teacher_samples}")
    print(f"Temperature: {temperature}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Generate unlabeled examples
    # ------------------------------------------------------------
    examples = generate_examples(
        workflow=workflow,
        num_examples=num_examples,
        seed=seed,
    )

    print(f"Generated {len(examples)} unlabeled examples")

    # ------------------------------------------------------------
    # Real teacher
    # ------------------------------------------------------------
    teacher = GroqTeacher(model=teacher_model)

    labeled_examples = []

    for index, example in enumerate(examples, start=1):
        print(
            f"[{index}/{len(examples)}] "
            f"Labeling example {example['id']}"
        )

        # Label all five decisions for this state.
        for question_name in example["questions"]:
            print(f"  -> {question_name}")

            example = label_example(
                teacher=teacher,
                example=example,
                question_name=question_name,
                num_samples=num_teacher_samples,
                temperature=temperature,
            )

        labeled_examples.append(example)

    # ------------------------------------------------------------
    # Validate before saving
    # ------------------------------------------------------------
    audit_examples(labeled_examples)

    output_path = output_dir / f"{split}.jsonl"

    save_jsonl(
        labeled_examples,
        output_path,
    )

    print()
    print(f"Saved: {output_path}")
    print(f"Examples: {len(labeled_examples)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and teacher-label Lyra synthetic data."
    )

    parser.add_argument(
        "--num-train",
        type=int,
        default=100,
        help="Number of training examples to generate.",
    )

    parser.add_argument(
        "--num-val",
        type=int,
        default=20,
        help="Number of validation examples to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--workflow",
        type=str,
        choices=WORKFLOWS,
        default="customer_service",
        help="Workflow to generate.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory where JSONL files are written.",
    )

    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help=(
            "Groq teacher model. If omitted, uses "
            "TEACHER_MODEL from .env."
        ),
    )

    parser.add_argument(
        "--teacher-samples",
        type=int,
        default=3,
        help="Number of teacher samples to average per decision.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Teacher sampling temperature.",
    )

    args = parser.parse_args()

    if args.num_train < 0:
        raise ValueError("--num-train must be >= 0")

    if args.num_val < 0:
        raise ValueError("--num-val must be >= 0")

    if args.teacher_samples < 1:
        raise ValueError("--teacher-samples must be >= 1")

    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be between 0 and 2")

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    if args.num_train > 0:
        generate_split(
            num_examples=args.num_train,
            split="train",
            workflow=args.workflow,
            seed=args.seed,
            output_dir=args.output_dir,
            teacher_model=args.teacher_model,
            num_teacher_samples=args.teacher_samples,
            temperature=args.temperature,
        )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    if args.num_val > 0:
        generate_split(
            num_examples=args.num_val,
            split="validation",
            workflow=args.workflow,
            seed=args.seed + 1_000_000,
            output_dir=args.output_dir,
            teacher_model=args.teacher_model,
            num_teacher_samples=args.teacher_samples,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()