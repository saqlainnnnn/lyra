from __future__ import annotations

import json
import os

import torch
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class Teacher:
    def predict_distribution(
        self,
        example: dict,
        question_name: str,
        temperature: float = 0.7,
    ) -> list[float]:
        raise NotImplementedError


def get_option_names(question: dict) -> list[str]:
    criteria = question["criteria"]

    if isinstance(criteria, dict):
        return list(criteria.keys())

    if isinstance(criteria, list):
        return [
            str(index)
            for index in range(len(criteria))
        ]

    raise TypeError(
        f"Unsupported criteria type: "
        f"{type(criteria).__name__}"
    )


def get_option_descriptions(
    question: dict,
) -> dict[str, str]:

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
        f"Unsupported criteria type: "
        f"{type(criteria).__name__}"
    )


class MockTeacher(Teacher):
    """
    Temporary teacher used for pipeline testing.
    """

    def predict_distribution(
        self,
        example: dict,
        question_name: str,
        temperature: float = 0.7,
    ) -> list[float]:

        question = example["questions"][
            question_name
        ]

        options = get_option_names(question)

        probabilities = torch.ones(
            len(options),
            dtype=torch.float32,
        )

        probabilities /= probabilities.sum()

        return probabilities.tolist()


class GroqTeacher(Teacher):
    """
    Real teacher backed by Groq.

    The model receives:
        - generated state
        - question
        - candidate options

    It does NOT receive the latent factors.
    """

    def __init__(
        self,
        model: str | None = None,
    ):

        api_key = os.environ.get(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set."
            )

        self.model = (
            model
            or os.environ.get(
                "TEACHER_MODEL",
                "openai/gpt-oss-120b",
            )
        )

        self.client = Groq(
            api_key=api_key
        )

    def _build_prompt(
        self,
        example: dict,
        question_name: str,
    ) -> str:

        question = example["questions"][
            question_name
        ]

        options = get_option_descriptions(
            question
        )

        state_json = json.dumps(
            example["state"],
            indent=2,
            ensure_ascii=False,
        )

        options_json = json.dumps(
            options,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are labeling a decision-making dataset.

You are given a scenario, a decision question,
and the allowed candidate answers.

Estimate a probability distribution over ALL
candidate answers.

The probabilities represent how plausible each
candidate is given the scenario.

Do not use information outside the scenario.

SCENARIO:

{state_json}

QUESTION:

{question["instructions"]}

QUESTION TYPE:

{question["type"]}

CANDIDATES:

{options_json}

Requirements:

- Return every candidate.
- Every probability must be between 0 and 1.
- Probabilities must sum to 1.
- Do not add candidates.
- Do not omit candidates.
- Treat score questions as ordered categories.
""".strip()

    def predict_distribution(
        self,
        example: dict,
        question_name: str,
        temperature: float = 0.7,
    ) -> list[float]:

        question = example["questions"][
            question_name
        ]

        options = get_option_names(
            question
        )

        prompt = self._build_prompt(
            example,
            question_name,
        )

        properties = {
            option: {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            }
            for option in options
        }

        schema = {
            "type": "object",
            "properties": properties,
            "required": options,
            "additionalProperties": False,
        }

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful decision "
                            "dataset labeling model. "
                            "Return probability distributions "
                            "over the provided candidates."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "decision_distribution",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        result = json.loads(content)

        probabilities = torch.tensor(
            [
                float(result[option])
                for option in options
            ],
            dtype=torch.float32,
        )

        if not torch.isfinite(
            probabilities
        ).all():

            raise ValueError(
                "Teacher returned non-finite "
                "probabilities."
            )

        if (probabilities < 0).any():

            raise ValueError(
                "Teacher returned a negative "
                "probability."
            )

        total = probabilities.sum()

        if total <= 0:

            raise ValueError(
                "Teacher probabilities sum to zero."
            )

        probabilities /= total

        return probabilities.tolist()


def average_distributions(
    distributions: list[list[float]],
) -> list[float]:

    if not distributions:
        raise ValueError(
            "No teacher distributions provided."
        )

    tensor = torch.tensor(
        distributions,
        dtype=torch.float32,
    )

    mean = tensor.mean(dim=0)

    mean /= mean.sum()

    return mean.tolist()


def label_example(
    teacher: Teacher,
    example: dict,
    question_name: str,
    num_samples: int = 3,
    temperature: float = 0.7,
) -> dict:

    distributions = []

    for _ in range(num_samples):

        distribution = (
            teacher.predict_distribution(
                example=example,
                question_name=question_name,
                temperature=temperature,
            )
        )

        distributions.append(
            distribution
        )

    gold = average_distributions(
        distributions
    )

    if "gold" not in example:
        example["gold"] = {}

    options = get_option_names(
        example["questions"][
            question_name
        ]
    )

    example["gold"][question_name] = {
        "probabilities": {
            option: probability
            for option, probability in zip(
                options,
                gold,
            )
        }
    }

    return example