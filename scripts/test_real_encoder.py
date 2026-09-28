import torch

from lyra.data.dataset import load_typed_decisions, parse_example
from lyra.data.format import format_decision_input
from lyra.data.tokenize import LyraTokenizer
from lyra.model.encoder import TransformerEncoder


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

    input_ids = encoded["input_ids"].unsqueeze(0)

    attention_mask = encoded["attention_mask"].unsqueeze(0)

    marker_positions = encoded["marker_positions"]

    print("input ids:")
    print(input_ids.shape)

    print("\nattention mask:")
    print(attention_mask.shape)

    print("\nmarker positions:")
    print(marker_positions)

    print("\noption names:")
    print(encoded["option_names"])

    encoder = TransformerEncoder(
        vocab_size=50368,
        hidden_size=192,
        num_layers=6,
        num_heads=3,
        intermediate_size=768,
    )

    output = encoder(
        input_ids,
        attention_mask,
    )

    print("\nencoder output:")
    print(output.shape)

    marker_hidden = output[
        0,
        marker_positions,
    ]

    print("\nmarker hidden states:")
    print(marker_hidden.shape)

    print("\nrequires gradient:")
    print(marker_hidden.requires_grad)


if __name__ == "__main__":
    main()