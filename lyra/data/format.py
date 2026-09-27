def format_state(state: dict) -> str:
    """Convert structured state into deterministic readable text."""

    lines = []

    account = state.get("account", {})

    for key, value in account.items():
        lines.append(f"account {key} {value}")

    thread = state.get("thread", [])

    for message in thread:
        role = message["role"]
        text = message["text"]
        lines.append(f"{role}: {text}")

    return "\n".join(lines)


def format_decision_input(example: dict, question_name: str) -> dict:
    """Format one parsed example into a Laya-style decision sequence."""

    if question_name not in example["questions"]:
        raise ValueError(
            f"Unknown question: {question_name}. "
            f"Available: {list(example['questions'])}"
        )

    question = example["questions"][question_name]

    question_type = question["type"]
    instruction = question["instructions"]
    criteria = question["criteria"]

    if isinstance(criteria, dict):
        options = list(criteria.items())
    else:
        options = [
            (str(i), description)
            for i, description in enumerate(criteria)
        ]

    parts = []

    # Question
    parts.append(f"<{question_type}>")
    parts.append(f"question: {instruction}")
    parts.append("[SEP]")

    # Options
    for option_name, description in options:
        parts.append(f"[MASK] {option_name}: {description}")

    parts.append("[SEP]")

    # State
    parts.append("<state>")
    parts.append(format_state(example["state"]))

    return {
        "text": " ".join(parts),
        "option_names": [name for name, _ in options],
        "question_type": question_type,
    }