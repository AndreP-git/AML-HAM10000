import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """Run one pass. If ``optimizer`` is None this is an eval pass."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in tqdm(loader, leave=False, disable=None):
        images, targets = images.to(device), targets.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    metrics = {
        "loss": total_loss / len(preds),
        "acc": float((preds == targets).mean()),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(targets, preds),
    }
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    run_name,
    logs_dir="logs",
    models_dir="models",
    amp=True,
    early_stopping_patience=7,
    scheduler=None,
):
    """Full training loop. Returns the path of the best checkpoint.
    """
    device = torch.device(device)
    model.to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp) if amp else None

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / f"{run_name}.csv"
    ckpt_path = Path(models_dir) / f"{run_name}.pt"

    fieldnames = [
        "epoch", "lr",
        "train_loss", "train_acc", "train_macro_f1", "train_balanced_acc",
        "val_loss", "val_acc", "val_macro_f1", "val_balanced_acc",
        "seconds",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = _run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        val_m = _run_epoch(model, val_loader, criterion, device, amp=amp)

        if scheduler is not None:
            scheduler.step(val_m["loss"])
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "lr": lr,
            "train_loss": train_m["loss"], "train_acc": train_m["acc"],
            "train_macro_f1": train_m["macro_f1"], "train_balanced_acc": train_m["balanced_acc"],
            "val_loss": val_m["loss"], "val_acc": val_m["acc"],
            "val_macro_f1": val_m["macro_f1"], "val_balanced_acc": val_m["balanced_acc"],
            "seconds": round(time.time() - t0, 1),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        marker = ""
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_macro_f1": best_f1, "run_name": run_name},
                ckpt_path,
            )
            marker = "  <- best (saved)"
        else:
            epochs_without_improvement += 1

        print(
            f"[{run_name}] epoch {epoch:2d}/{epochs} "
            f"train_loss={train_m['loss']:.3f} "
            f"val_loss={val_m['loss']:.3f} "
            f"val_macroF1={val_m['macro_f1']:.3f} "
            f"val_balAcc={val_m['balanced_acc']:.3f}{marker}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"[{run_name}] early stopping at epoch {epoch} "
                  f"(no val macro-F1 gain for {early_stopping_patience} epochs)")
            break

    print(f"[{run_name}] best val macro-F1 = {best_f1:.4f}  ->  {ckpt_path}")
    return ckpt_path
