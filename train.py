import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    HAM10000Dataset,
    compute_class_weights,
    make_weighted_sampler,
)
from src.engine import train_model
from src.models import build_model
from src.transforms import build_transforms, normalization_for_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default=None, help="simple_cnn | resnet18")
    p.add_argument("--imbalance", default=None, choices=["none", "weighted_loss", "oversampling"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # ---- settings -------------------------
    model_name = args.model or cfg["model"]["name"]
    imbalance = args.imbalance or cfg["imbalance"]
    epochs = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr or cfg["train"]["lr"]
    freeze_backbone = args.freeze_backbone or cfg["model"]["freeze_backbone"]
    run_name = args.run_name or f"{model_name}_{imbalance}"

    torch.manual_seed(cfg["train"]["seed"])

    image_size = cfg["data"]["image_size"]
    dataset_mean = cfg["data"]["normalization"]["mean"]
    dataset_std = cfg["data"]["normalization"]["std"]
    mean, std = normalization_for_model(model_name, dataset_mean, dataset_std)
    train_tf, eval_tf = build_transforms(image_size, mean, std)

    splits_csv = cfg["data"]["splits_csv"]
    images_dir = Path(cfg["data"]["raw_dir"]) / "images"

    train_ds = HAM10000Dataset(splits_csv, images_dir, "train", train_tf)
    val_ds = HAM10000Dataset(splits_csv, images_dir, "val", eval_tf)

    # ---- imbalance strategy ---------------------------------------------
    sampler, shuffle, class_weights = None, True, None
    if imbalance == "oversampling":
        sampler = make_weighted_sampler(train_ds.labels)
        shuffle = False  # sampler and shuffle are mutually exclusive
    elif imbalance == "weighted_loss":
        class_weights = compute_class_weights(train_ds.labels).to(args.device)

    num_workers = cfg["train"]["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- model ---------------------------------------
    model = build_model(
        model_name, pretrained=cfg["model"]["pretrained"], freeze_backbone=freeze_backbone
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["train"]["scheduler_patience"],
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Run: {run_name} | model={model_name} imbalance={imbalance} "
          f"| trainable params={n_trainable:,} | device={args.device}")
    print(f"train={len(train_ds)} val={len(val_ds)} images")

    train_model(
        model, train_loader, val_loader, criterion, optimizer,
        device=args.device, epochs=epochs, run_name=run_name,
        logs_dir=cfg["paths"]["logs_dir"], models_dir=cfg["paths"]["models_dir"],
        amp=cfg["train"]["amp"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
