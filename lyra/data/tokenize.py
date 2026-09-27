from transformers import AutoTokenizer


MODEL_NAME = "answerdotai/ModernBERT-base"


class LyraTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_decision(self, formatted: dict) -> dict:
        """
        Convert a formatted decision into tensors consumed by Lyra.
        """

        encoded = self.tokenizer(
            formatted["text"],
            return_tensors="pt",
            padding=False,
            truncation=True,
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        mask_token_id = self.tokenizer.mask_token_id

        marker_positions = (
            input_ids == mask_token_id
        ).nonzero(as_tuple=True)[0]

        expected_markers = len(formatted["option_names"])

        if len(marker_positions) != expected_markers:
            raise ValueError(
                f"Expected {expected_markers} marker tokens, "
                f"but found {len(marker_positions)}."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_positions": marker_positions,
            "option_names": formatted["option_names"],
            "question_type": formatted["question_type"],
        }