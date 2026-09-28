import torch


def collate_decisions(batch):
    """
    Collate tokenized decision examples with:

        - variable sequence lengths
        - variable numbers of candidates

    Returns:
        input_ids:
            [B, max_seq_len]

        attention_mask:
            [B, max_seq_len]

        marker_positions:
            [B, max_candidates]

        candidate_mask:
            [B, max_candidates]

        targets:
            [B, max_candidates]

        option_names:
            Python list of option-name lists

        question_types:
            Python list of question types
    """

    batch_size = len(batch)

    max_seq_len = max(
        item["input_ids"].shape[0]
        for item in batch
    )

    max_candidates = max(
        item["marker_positions"].shape[0]
        for item in batch
    )

    input_ids = torch.zeros(
        batch_size,
        max_seq_len,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        batch_size,
        max_seq_len,
        dtype=torch.long,
    )

    marker_positions = torch.zeros(
        batch_size,
        max_candidates,
        dtype=torch.long,
    )

    candidate_mask = torch.zeros(
        batch_size,
        max_candidates,
        dtype=torch.bool,
    )

    targets = torch.zeros(
        batch_size,
        max_candidates,
        dtype=torch.float32,
    )

    option_names = []
    question_types = []

    for i, item in enumerate(batch):

        seq_len = item["input_ids"].shape[0]
        num_candidates = item["marker_positions"].shape[0]

        input_ids[i, :seq_len] = item["input_ids"]

        attention_mask[i, :seq_len] = item["attention_mask"]

        marker_positions[i, :num_candidates] = (
            item["marker_positions"]
        )

        candidate_mask[i, :num_candidates] = True

        targets[i, :num_candidates] = item["target"]

        option_names.append(
            item["option_names"]
        )

        question_types.append(
            item["question_type"]
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_positions": marker_positions,
        "candidate_mask": candidate_mask,
        "targets": targets,
        "option_names": option_names,
        "question_types": question_types,
    }