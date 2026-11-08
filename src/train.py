"""
Training script for Dual-Head Siamese U-Net on xView2.
Usage:
    python -m src.train --epochs 20 --batch-size 4 --lr 2e-4
"""

import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import (
    CHECKPOINT_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    DEVICE,
    OUTPUT_ROOT,
    RANDOM_STATE,
    USE_AMP,
)
from src.dataset import get_dataloaders
from src.loss import DualHeadLoss
from src.metrics import calculate_metrics_from_confusion, compute_confusion_matrix
from src.model import DualHeadSiameseUNet


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_loc_loss = 0.0
    total_dmg_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        loc_tgt = batch["loc_target"].to(device, non_blocking=True)
        dmg_tgt = batch["damage_target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP, dtype=torch.float16):
            loc_logits, dmg_logits = model(pre_img, post_img)
            loss_dict = criterion(loc_logits, dmg_logits, loc_tgt, dmg_tgt)
            loss = loss_dict["total_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_loc_loss += loss_dict["loss_loc"].item()
        total_dmg_loss += loss_dict["loss_dmg"].item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            loc=f"{loss_dict['loss_loc'].item():.3f}",
            dmg=f"{loss_dict['loss_dmg'].item():.3f}",
        )

    num_batches = step + 1 if max_batches is not None else len(loader)
    return {
        "loss": total_loss / max(num_batches, 1),
        "loc_loss": total_loc_loss / max(num_batches, 1),
        "dmg_loss": total_dmg_loss / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    confusion = np.zeros((5, 5), dtype=np.int64)

    pbar = tqdm(loader, desc="Validation", leave=False)
    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        loc_tgt = batch["loc_target"].to(device, non_blocking=True)
        dmg_tgt = batch["damage_target"].to(device, non_blocking=True)
        raw_tgt = batch["raw_target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP, dtype=torch.float16):
            loc_logits, dmg_logits = model(pre_img, post_img)
            loss_dict = criterion(loc_logits, dmg_logits, loc_tgt, dmg_tgt)

            # Generate combined prediction
            loc_probs = torch.sigmoid(loc_logits).squeeze(1)
            dmg_preds = torch.argmax(dmg_logits, dim=1) + 1  # 0..3 -> 1..4
            final_pred = torch.where(loc_probs >= 0.5, dmg_preds, torch.zeros_like(dmg_preds))

        total_loss += loss_dict["total_loss"].item()
        confusion += compute_confusion_matrix(final_pred, raw_tgt)

    metrics = calculate_metrics_from_confusion(confusion)
    num_eval = step + 1 if max_batches is not None else len(loader)
    metrics["val_loss"] = total_loss / max(num_eval, 1)
    return metrics


def plot_and_save_history(history: dict, save_path: Path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="crimson")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Localization F1
    axes[0, 1].plot(epochs, history["loc_f1"], label="Localization F1", color="forestgreen", marker="o")
    axes[0, 1].set_title("Building Localization F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Damage Macro F1 & mIoU
    axes[1, 0].plot(epochs, history["fg_macro_f1"], label="Damage Macro F1", color="darkorange", marker="s")
    axes[1, 0].plot(epochs, history["fg_miou"], label="Damage mIoU", color="purple", marker="^")
    axes[1, 0].set_title("Damage Assessment (Foreground Classes 1-4)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Official xView2 Combined Score
    axes[1, 1].plot(epochs, history["xview2_score"], label="xView2 Score (0.3*Loc + 0.7*Dmg)", color="teal", marker="d")
    axes[1, 1].set_title("Official xView2 Score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_training(
    epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    resume: str = None,
    backbone: str = "resnet34",
):
    print("=" * 70)
    print("TERRASHIELD AI — DUAL-HEAD TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone: {backbone.upper()}")
    print(f"Batch size: {batch_size} (512x512 building-guided crops)")
    print(f"Learning rate: {lr} | Weight decay: {weight_decay}")
    print(f"Max epochs: {epochs} | Early stopping patience: {patience}")
    print(f"Checkpoints: {CHECKPOINT_ROOT}")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    train_loader, hold_loader, _ = get_dataloaders(batch_size=batch_size)
    print(f"Train samples: {len(train_loader.dataset)} | Holdout samples: {len(hold_loader.dataset)}")

    model = DualHeadSiameseUNet(backbone=backbone, pretrained=True).to(DEVICE)
    criterion = DualHeadLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=0.002, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    tag = "siamese" if backbone == "resnet34" else f"efficientnet_b2"
    best_checkpoint_path = CHECKPOINT_ROOT / f"dual_head_{tag}_best.pth"
    latest_checkpoint_path = CHECKPOINT_ROOT / f"dual_head_{tag}_latest.pth"
    history_plot_path = OUTPUT_ROOT / f"dual_head_{tag}_training_history.png"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "loc_f1": [],
        "fg_macro_f1": [],
        "fg_miou": [],
        "xview2_score": [],
        "lr": [],
    }

    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1

    if resume:
        ckpt_path = Path(resume)
        if ckpt_path.exists():
            print(f"\nResuming training from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for g in optimizer.param_groups:
                    g["lr"] = lr
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "history" in checkpoint and isinstance(checkpoint["history"], dict):
                history = checkpoint["history"]
            if "best_score" in checkpoint:
                best_score = checkpoint["best_score"]
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_epoch = checkpoint.get("epoch", 0)
            print(f"Loaded successfully. Resuming from Epoch {start_epoch} with initial best score: {best_score:.4f}")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_res = evaluate_one_epoch(model, hold_loader, criterion, DEVICE)

        current_score = val_res["xview2_score"]
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(current_score)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["val_loss"])
        history["loc_f1"].append(val_res["loc_f1"])
        history["fg_macro_f1"].append(val_res["fg_macro_f1"])
        history["fg_miou"].append(val_res["fg_miou"])
        history["xview2_score"].append(current_score)
        history["lr"].append(current_lr)

        elapsed = (time.time() - t0) / 60.0
        print(
            f"Train Loss: {train_res['loss']:.4f} | Val Loss: {val_res['val_loss']:.4f} | "
            f"Loc F1: {val_res['loc_f1']:.4f} | Dmg F1: {val_res['fg_macro_f1']:.4f} | "
            f"xView2 Score: {current_score:.4f} | Time: {elapsed:.1f}m"
        )

        # Checkpoint saving
        is_best = current_score > best_score
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score if not is_best else current_score,
            "val_metrics": val_res,
            "history": history,
        }

        torch.save(checkpoint_data, latest_checkpoint_path)

        if is_best:
            best_score = current_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"[*] New best checkpoint saved to {best_checkpoint_path} (Score: {best_score:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience} (Best was Epoch {best_epoch}: {best_score:.4f})")

        plot_and_save_history(history, history_plot_path)

        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered: no improvement for {patience} epochs.")
            break

    total_time = (time.time() - start_time) / 3600.0
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {total_time:.2f} hours")
    print(f"Best Epoch: {best_epoch} | Best xView2 Score: {best_score:.4f}")
    print(f"Saved Checkpoint: {best_checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Head Siamese U-Net on xView2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet34", "efficientnet_b2"], help="Encoder backbone architecture")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        resume=args.resume,
        backbone=args.backbone,
    )

