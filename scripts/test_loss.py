import torch

from lyra.model.loss import soft_cross_entropy


def main():
    target = torch.tensor(
        [[0.743333, 0.030000, 0.200000, 0.003333, 0.023333]],
        dtype=torch.float32,
    )

    # These are the ONLY things being learned.
    logits = torch.nn.Parameter(torch.zeros(1, 5))

    optimizer = torch.optim.AdamW(
        [logits],
        lr=0.1,
    )

    for step in range(100):
        optimizer.zero_grad()

        loss = soft_cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)

            print(
                f"step {step:03d} "
                f"loss={loss.item():.6f} "
                f"probs={probs.squeeze(0).tolist()}"
            )


if __name__ == "__main__":
    main()