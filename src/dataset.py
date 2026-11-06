from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASSES)

CLASS_FULLNAME = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


class HAM10000Dataset(Dataset):
    """
    splits_csv : path to metadata_splits.csv
    images_dir : folder containing ``<image_id>.jpg`` files
    split      : one of 'train' | 'val' | 'test'
    transform  : a torchvision transform applied to each PIL image
    """

    def __init__(self, splits_csv, images_dir, split, transform=None):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no rows for split={split!r} in {splits_csv}")
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.labels = self.df["dx"].map(CLASS_TO_IDX).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / f"{row['image_id']}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights.

    Formula ``w_c = N / (C * n_c)`` where N is the total count and C the
    number of classes: rare classes get a large weight, the dominant class
    a small one.
    """
    labels = torch.as_tensor(labels)
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1)  # guard against an absent class
    weights = counts.sum() / (num_classes * counts)
    return weights


def make_weighted_sampler(labels, num_classes=NUM_CLASSES):
    """Build a WeightedRandomSampler that balances classes on average.

    Each *sample* is drawn with probability inversely proportional to its
    class frequency, so on average every class appears equally often. This
    is the oversampling strategy: minority images are repeated within an
    epoch instead of being reweighted in the loss.
    """
    labels = torch.as_tensor(labels)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    per_class_weight = 1.0 / counts
    sample_weights = per_class_weight[labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(labels), replacement=True
    )
