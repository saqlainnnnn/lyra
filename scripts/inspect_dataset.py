from lyra.data.dataset import load_typed_decisions, parse_example
from lyra.data.format import format_decision_input
from lyra.data.tokenize import LyraTokenizer


def main():
    dataset = load_typed_decisions(
        workflow="customer_service",
        split="train",
    )

    example = parse_example(dataset[0])

    formatted = format_decision_input(
        example,
        question_name="action",
    )

    tokenizer = LyraTokenizer()
    encoded = tokenizer.tokenize_decision(formatted)

    print("=" * 80)
    print("LYRA — PHASE 1 COMPLETE")
    print("=" * 80)

    print("\nQuestion type:")
    print(encoded["question_type"])

    print("\nOptions:")
    for i, option in enumerate(encoded["option_names"]):
        print(f"  {i}: {option}")

    print("\nTensor shapes:")
    print(f"  input_ids:        {tuple(encoded['input_ids'].shape)}")
    print(f"  attention_mask:   {tuple(encoded['attention_mask'].shape)}")
    print(f"  marker_positions: {tuple(encoded['marker_positions'].shape)}")

    print("\nSequence length:")
    print(encoded["input_ids"].shape[0])

    print("\nMarker positions:")
    print(encoded["marker_positions"].tolist())

    print("\nMarker → option:")
    for position, option in zip(
        encoded["marker_positions"].tolist(),
        encoded["option_names"],
    ):
        print(f"  {position:4d} → {option}")

    print("\nFirst 30 tokens:")
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )

    for i, token in enumerate(tokens[:30]):
        print(f"  {i:4d}  {token}")

    print("\nLast 10 tokens:")
    start = max(0, len(tokens) - 10)

    for i, token in enumerate(tokens[start:], start=start):
        print(f"  {i:4d}  {token}")

    print("\n" + "=" * 80)
    print("=" * 80)


if __name__ == "__main__":
    main()