import torch


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    if mask is not None:
        logits = logits.masked_fill(
            ~mask,
            torch.finfo(logits.dtype).min,
        )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    loss = -(
        targets * log_probs
    ).sum(dim=-1)

    return loss.mean()