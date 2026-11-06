import colorsys

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .dataset import CLASS_TO_IDX, NUM_CLASSES


def build_dummy(strategy="most_frequent", seed=42):
    """Return an (unfitted) sklearn DummyClassifier.

    strategy: "most_frequent" (majority class) or "stratified" (prior-based
    random guessing).
    """
    return DummyClassifier(strategy=strategy, random_state=seed)


def image_features(pil_image, bins=16, size=64):
    """Compact colour/texture descriptor for one image.

    Concatenates, over a downscaled copy:
      - per-channel RGB histograms (colour distribution),
      - per-channel HSV histograms (hue/saturation cues that matter for skin),
      - per-channel mean and std (overall tone / contrast).
    Returns a 1-D float32 vector; length = 6*bins + 6.
    """
    img = pil_image.convert("RGB").resize((size, size))
    rgb = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0

    feats = []
    for arr in (rgb, hsv):
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 1), density=True)
            feats.append(hist)
    feats.append(rgb.reshape(-1, 3).mean(axis=0))
    feats.append(rgb.reshape(-1, 3).std(axis=0))
    return np.concatenate(feats).astype(np.float32)


def extract_feature_matrix(df_split, images_dir, bins=16, size=64, show_progress=True):
    """Build (X, y) for one split's dataframe (rows of metadata_splits.csv)."""
    from pathlib import Path

    images_dir = Path(images_dir)
    rows = df_split.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(df_split), desc="features")
    X, y = [], []
    for row in rows:
        img = Image.open(images_dir / f"{row.image_id}.jpg")
        X.append(image_features(img, bins=bins, size=size))
        y.append(CLASS_TO_IDX[row.dx])
    return np.stack(X), np.asarray(y)


def build_classical(kind="logreg", seed=42):
    """Return an (unfitted) classical classifier.

    kind: "logreg" (StandardScaler + multinomial Logistic Regression, with
    balanced class weights) or "rf" (Random Forest, balanced). Both expose
    predict_proba for AUC.
    """
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    raise ValueError(f"unknown classical baseline kind: {kind!r}")


def predict_proba_aligned(clf, X):
    """predict_proba but always shaped (N, NUM_CLASSES).
    """
    proba = clf.predict_proba(X)
    if proba.shape[1] == NUM_CLASSES:
        return proba
    full = np.zeros((X.shape[0], NUM_CLASSES), dtype=proba.dtype)
    for col, cls in enumerate(clf.classes_):
        full[:, cls] = proba[:, col]
    return full
