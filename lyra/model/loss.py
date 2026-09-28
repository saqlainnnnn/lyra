import torch


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy between a predicted distribution and
    a soft target distribution.

    logits:
        [batch_size, num_options]

    targets:
        [batch_size, num_options]
    """

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    loss = -(targets * log_probs).sum(
        dim=-1
    )

    return loss.mean()