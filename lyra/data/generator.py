from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


DATASET_NAME = "LocalLLaMA/typed-decisions"

WORKFLOWS = [
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
]


@dataclass
class GenerationTemplate:
    workflow: str
    questions: dict[str, Any]
    factor_examples: list[dict[str, Any]]


def load_generation_template(
    workflow: str,
    split: str = "train",
) -> GenerationTemplate:
    """
    Load the official typed-decisions dataset and use it as the
    schema/template source.

    We extract:
      - official question definitions
      - official factor examples

    We do NOT reuse the official state or gold labels.
    """

    if workflow not in WORKFLOWS:
        raise ValueError(
            f"Unknown workflow: {workflow}. "
            f"Available: {WORKFLOWS}"
        )

    dataset = load_dataset(
        DATASET_NAME,
        workflow,
        split=split,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No examples found for workflow={workflow}"
        )

    first = dataset[0]

    questions = json.loads(
        first["questions"]
    )

    factor_examples = [
        json.loads(row["factors"])
        for row in dataset
    ]

    return GenerationTemplate(
        workflow=workflow,
        questions=questions,
        factor_examples=factor_examples,
    )


def sample_factors(
    template: GenerationTemplate,
    rng: random.Random,
    seed: int,
) -> dict[str, Any]:
    """
    Sample factor values from values that actually occur in the
    official dataset.

    This prevents us from inventing a completely unrelated factor
    vocabulary.
    """

    examples = template.factor_examples

    keys = set()

    for example in examples:
        keys.update(example.keys())

    factors = {}

    for key in sorted(keys):

        values = []

        for example in examples:

            if key not in example:
                continue

            value = example[key]

            # We currently support scalar categorical/numeric factors.
            if isinstance(value, (str, int, float, bool)):
                values.append(value)

        if not values:
            continue

        factors[key] = rng.choice(values)

    factors["workflow"] = template.workflow
    factors["generation_seed"] = seed

    return factors


def render_customer_service_state(
    factors: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """
    Render customer-service latent factors into a richer
    synthetic conversation state.

    The factor vocabulary comes from the official dataset.
    The actual wording and rendering are our Lyra reproduction,
    not the original private renderer.
    """

    topic = factors.get(
        "topic",
        "general customer service",
    )

    tone = factors.get(
        "tone",
        "neutral",
    )

    amount_usd = factors.get(
        "amount_usd",
        0,
    )

    has_deadline = factors.get(
        "has_deadline",
        False,
    )

    money_involved = factors.get(
        "money_involved",
        False,
    )

    prior_resolution = factors.get(
        "prior_resolution",
        "none",
    )

    prior_tickets = factors.get(
        "prior_tickets_90d",
        0,
    )

    tenure_months = factors.get(
        "tenure_months",
        0,
    )

    text_generated = factors.get(
        "text_generated",
        True,
    )

    threatens_to_leave = factors.get(
        "threatens_to_leave",
        False,
    )

    tier = factors.get(
        "tier",
        "standard",
    )

    turns = factors.get(
        "turns",
        1,
    )

    # ---------------------------------------------------------
    # Customer message
    # ---------------------------------------------------------

    message_templates = {
        "neutral": [
            (
                f"I need help with my {topic}. "
                "Can you tell me what I should do?"
            ),
            (
                f"I'm contacting you about a {topic} issue. "
                "Could you help me resolve it?"
            ),
        ],
        "confused": [
            (
                f"I'm not sure what happened with my {topic}. "
                "Can you explain what is going on?"
            ),
            (
                f"I don't understand this {topic} issue. "
                "Can you help me figure it out?"
            ),
        ],
        "frustrated": [
            (
                f"I'm getting frustrated with this {topic} problem. "
                "I need some help resolving it."
            ),
            (
                f"This {topic} issue has been frustrating. "
                "Can someone help me sort it out?"
            ),
        ],
        "angry": [
            (
                f"I've had enough of this {topic} issue. "
                "I need this resolved."
            ),
            (
                f"This is unacceptable. "
                f"I need my {topic} problem fixed."
            ),
        ],
        "positive": [
            (
                f"Hi, I have a question about my {topic}. "
                "Could you help me?"
            ),
            (
                f"Thanks for your help. "
                f"I need some assistance with my {topic}."
            ),
        ],
    }

    user_message = rng.choice(
        message_templates.get(
            str(tone),
            message_templates["neutral"],
        )
    )

    # ---------------------------------------------------------
    # Account context
    # ---------------------------------------------------------

    account = {
        "tier": tier,
        "tenure_months": tenure_months,
        "prior_tickets_90d": prior_tickets,
    }

    # ---------------------------------------------------------
    # System context
    # ---------------------------------------------------------

    context_parts = [
        f"Topic: {topic}.",
        f"Tone: {tone}.",
        f"Amount involved: ${amount_usd}.",
        f"Money involved: {money_involved}.",
        f"Deadline: {has_deadline}.",
        f"Previous resolution status: {prior_resolution}.",
        f"Previous tickets in the last 90 days: {prior_tickets}.",
        f"Customer tenure: {tenure_months} months.",
        f"Customer tier: {tier}.",
        f"Threatens to leave: {threatens_to_leave}.",
        f"Conversation turns so far: {turns}.",
    ]

    if text_generated:
        context_parts.append(
            "The conversation text was generated from the "
            "underlying scenario."
        )

    system_context = " ".join(
        context_parts
    )

    # ---------------------------------------------------------
    # Previous resolution information
    # ---------------------------------------------------------

    if prior_resolution == "none":
        resolution_message = (
            "There is no previous resolution recorded for this issue."
        )
    else:
        resolution_message = (
            f"A previous resolution was recorded as "
            f"'{prior_resolution}'."
        )

    return {
        "account": account,

        "thread": [
            {
                "role": "system",
                "text": system_context,
            },
            {
                "role": "user",
                "text": user_message,
            },
            {
                "role": "system",
                "text": resolution_message,
            },
        ],
    }


def render_state(
    workflow: str,
    factors: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """
    Dispatch to a workflow-specific renderer.
    """

    if workflow == "customer_service":
        return render_customer_service_state(
            factors=factors,
            rng=rng,
        )

    raise NotImplementedError(
        f"No synthetic renderer exists yet for "
        f"workflow={workflow}"
    )


def build_example(
    template: GenerationTemplate,
    rng: random.Random,
    seed: int,
) -> dict[str, Any]:

    factors = sample_factors(
        template=template,
        rng=rng,
        seed=seed,
    )

    state = render_state(
        workflow=template.workflow,
        factors=factors,
        rng=rng,
    )

    return {
        "id": str(uuid.uuid4()),
        "workflow": template.workflow,
        "split": "generated",
        "state": state,

        # IMPORTANT:
        # These questions come directly from the official dataset.
        "questions": template.questions,

        # Factors are retained for auditing only.
        "factors": factors,
    }


def generate_examples(
    num_examples: int,
    workflow: str,
    seed: int,
) -> list[dict[str, Any]]:

    template = load_generation_template(
        workflow=workflow,
    )

    rng = random.Random(seed)

    examples = []

    for index in range(num_examples):

        example_seed = seed + index

        example = build_example(
            template=template,
            rng=rng,
            seed=example_seed,
        )

        examples.append(example)

    return examples