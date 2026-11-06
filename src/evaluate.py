import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .dataset import CLASSES


@torch.no_grad()
def predict(model, loader, device):
    """Return ``(y_true, y_pred, y_prob)`` over an entire loader."""
    device = torch.device(device)
    model.to(device).eval()
    trues, preds, probs = [], [], []
    for images, targets in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
        trues.append(targets.numpy())
    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def load_checkpoint(model, ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # a class absent from y_true breaks ovr AUC
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_auc_ovr": auc,
    }


def per_class_report(y_true, y_pred):
    return classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )


def confusion(y_true, y_pred, normalize=None):
    return confusion_matrix(
        y_true, y_pred, labels=range(len(CLASSES)), normalize=normalize
    )
