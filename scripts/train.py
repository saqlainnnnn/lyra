import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lyra.data.collate import collate_decisions
from lyra.data.dataset import DecisionDataset
from lyra.data.tokenize import LyraTokenizer
from lyra.model.loss import soft_cross_entropy
from lyra.model.lyra import Lyra


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    return (
        checkpoint["epoch"],
        checkpoint["global_step"],
        checkpoint["best_val_loss"],
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    global_step,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        dataloader,
        desc=f"Train {epoch}",
    )

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        marker_positions = batch["marker_positions"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_positions=marker_positions,
        )

        loss = soft_cross_entropy(
            logits,
            targets,
            mask=candidate_mask,
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = input_ids.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        global_step += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            grad=f"{grad_norm:.2f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    epoch_loss = total_loss / total_examples

    return epoch_loss, global_step


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        dataloader,
        desc="Validation",
    )

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        marker_positions = batch["marker_positions"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        targets = batch["targets"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_positions=marker_positions,
        )

        loss = soft_cross_entropy(
            logits,
            targets,
            mask=candidate_mask,
        )

        batch_size = input_ids.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_examples


def build_dataloaders(
    workflow,
    question,
    batch_size,
    val_fraction=0.1,
    seed=42,
):
    tokenizer = LyraTokenizer()

    dataset = DecisionDataset(
        workflow=workflow,
        split="train",
        question_name=question,
        tokenizer=tokenizer,
    )

    val_size = max(
        1,
        int(len(dataset) * val_fraction),
    )

    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_decisions,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_decisions,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def build_model(config):
    model_config = config["model"]
    decision_config = config["decision_head"]

    model = Lyra(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        intermediate_size=model_config["intermediate_size"],
        decision_num_layers=decision_config["num_layers"],
        decision_num_heads=decision_config["num_heads"],
        decision_intermediate_size=decision_config["intermediate_size"],
        dropout=model_config["dropout"],
    )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workflow",
        type=str,
        default="customer_service",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="action",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    train_loader, val_loader = build_dataloaders(
        workflow=args.workflow,
        question=args.question,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    model = build_model(config)
    model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {total_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.checkpoint is not None:
        print(
            f"Loading checkpoint: {args.checkpoint}"
        )

        (
            checkpoint_epoch,
            global_step,
            best_val_loss,
        ) = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            scheduler,
        )

        start_epoch = checkpoint_epoch + 1

        print(
            f"Resuming from epoch {start_epoch}"
        )

        print(
            f"Global step: {global_step}"
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.6f}"
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
        )
        print(
            f"  train loss: {train_loss:.6f}"
        )
        print(
            f"  val loss:   {val_loss:.6f}"
        )
        print(
            f"  best val:   {best_val_loss:.6f}"
        )
        print(
            f"  lr:         {current_lr:.6e}"
        )
        print(
            f"  step:       {global_step}"
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            best_val_loss,
        )

        print(
            f"Saved: {latest_path}"
        )

        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_val_loss,
            )

            print(
                f"New best validation loss: "
                f"{best_val_loss:.6f}"
            )

            print(
                f"Saved: {best_path}"
            )

        print()


if __name__ == "__main__":
    main()