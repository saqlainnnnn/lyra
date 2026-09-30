from __future__ import annotations

import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class Teacher:
    def predict_distribution(
        self,
        example,
        question_name: str,
        temperature: float = 0.7,
    ) -> list[float]:
        raise NotImplementedError


def get_option_names(question):
    criteria = question["criteria"]

    if isinstance(criteria, dict):
        return list(criteria.keys())

    if isinstance(criteria, list):
        return [str(index) for index in range(len(criteria))]

    raise TypeError(
        f"Unsupported criteria type: {type(criteria)}"
    )


def get_option_descriptions(question):
    criteria = question["criteria"]

    if isinstance(criteria, dict):
        return {
            str(name): str(description)
            for name, description in criteria.items()
        }

    if isinstance(criteria, list):
        return {
            str(index): str(description)
            for index, description in enumerate(criteria)
        }

    raise TypeError(
        f"Unsupported criteria type: {type(criteria)}"
    )


class LocalTeacher(Teacher):

    def __init__(
        self,
        model,
        tokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def _build_prompt(
        self,
        example,
        question_name: str,
    ) -> str:

        question = example["questions"][question_name]

        options = get_option_names(question)
        descriptions = get_option_descriptions(question)

        candidate_text = "\n".join(
            f"- {option}: {descriptions[option]}"
            for option in options
        )

        return f"""
You are a decision labeling model.

Analyze the provided state and estimate the probability distribution
over the candidate decisions.

The probabilities represent your uncertainty about which candidate
is appropriate.

STATE:
{example["state"]}

QUESTION:
{question_name}

QUESTION TYPE:
{question["type"]}

CANDIDATES:
{candidate_text}

Return ONLY valid JSON.

The JSON must contain exactly these keys:
{json.dumps(options)}

Each value must be a number between 0 and 1.

The probabilities must sum to 1.

Do not add explanations.
Do not add markdown.
"""

    def predict_distribution(
        self,
        example,
        question_name: str,
        temperature: float = 0.7,
    ) -> list[float]:

        options = get_option_names(
            example["questions"][question_name]
        )

        prompt = self._build_prompt(
            example,
            question_name,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a decision labeling model. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
            )

        generated_tokens = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        try:
            result = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Teacher returned invalid JSON:\n{response}"
            ) from exc

        expected_keys = set(options)
        actual_keys = set(result)

        if actual_keys != expected_keys:
            raise ValueError(
                "Teacher returned incorrect candidate keys. "
                f"Expected: {sorted(expected_keys)} "
                f"Got: {sorted(actual_keys)}"
            )

        probabilities = torch.tensor(
            [float(result[option]) for option in options],
            dtype=torch.float32,
        )

        if not torch.isfinite(probabilities).all():
            raise ValueError(
                f"Teacher returned non-finite probabilities: "
                f"{probabilities}"
            )

        if (probabilities < 0).any():
            raise ValueError(
                f"Teacher returned negative probabilities: "
                f"{probabilities}"
            )

        total = probabilities.sum()

        if total <= 0:
            raise ValueError(
                f"Teacher returned zero probability mass: "
                f"{probabilities}"
            )

        probabilities /= total

        return probabilities.tolist()


def average_distributions(
    distributions: list[list[float]],
) -> list[float]:

    probabilities = torch.tensor(
        distributions,
        dtype=torch.float32,
    )

    probabilities = probabilities.mean(dim=0)

    probabilities /= probabilities.sum()

    return probabilities.tolist()


def label_example(
    teacher: Teacher,
    example,
    question_name: str,
    num_samples: int = 3,
    temperature: float = 0.7,
):

    distributions = []

    for _ in range(num_samples):
        distribution = teacher.predict_distribution(
            example=example,
            question_name=question_name,
            temperature=temperature,
        )

        distributions.append(distribution)

    averaged = average_distributions(
        distributions
    )

    options = get_option_names(
        example["questions"][question_name]
    )

    if "gold" not in example:
        example["gold"] = {}

    example["gold"][question_name] = {
        "probabilities": {
            option: probability
            for option, probability in zip(
                options,
                averaged,
            )
        }
    }

    return example