import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

N_IMAGES_EXPECTED = 10015


def extract_archives(raw_dir: Path) -> None:
    """Extract every zip in raw_dir"""
    for zip_path in sorted(raw_dir.glob("*.zip")):
        marker = raw_dir / f".extracted_{zip_path.stem}"
        if marker.exists():
            print(f"[skip] {zip_path.name} already extracted")
            continue
        print(f"[extract] {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        marker.touch()


def collect_images(raw_dir: Path) -> Path:
    """Merge all lesion jpgs into raw_dir/images/ and return that folder."""
    images_dir = raw_dir / "images"
    images_dir.mkdir(exist_ok=True)

    n_existing = len(list(images_dir.glob("*.jpg")))
    if n_existing >= N_IMAGES_EXPECTED:
        print(f"[skip] {n_existing} images already in {images_dir}")
        return images_dir

    moved = 0
    for jpg in raw_dir.rglob("ISIC_*.jpg"):
        if jpg.parent == images_dir:
            continue
        target = images_dir / jpg.name
        if not target.exists():
            shutil.move(str(jpg), target)
            moved += 1
    total = len(list(images_dir.glob("*.jpg")))
    print(f"[images] moved {moved}, total {total} in {images_dir}")
    return images_dir


def build_splits(
    metadata: pd.DataFrame, train_frac: float, val_frac: float, seed: int
) -> pd.DataFrame:
    """Assign each image to train/val/test, grouping by lesion_id.
    """
    lesions = metadata.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    train_lesions, rest = train_test_split(
        lesions,
        train_size=train_frac,
        stratify=lesions["dx"],
        random_state=seed,
    )
    # val_frac is expressed as a fraction of the whole dataset
    val_share_of_rest = val_frac / (1.0 - train_frac)
    val_lesions, test_lesions = train_test_split(
        rest,
        train_size=val_share_of_rest,
        stratify=rest["dx"],
        random_state=seed,
    )

    split_map = {}
    for name, subset in [("train", train_lesions), ("val", val_lesions), ("test", test_lesions)]:
        split_map.update(dict.fromkeys(subset["lesion_id"], name))

    metadata = metadata.copy()
    metadata["split"] = metadata["lesion_id"].map(split_map)
    return metadata


def sanity_checks(metadata: pd.DataFrame, images_dir: Path) -> None:
    assert len(metadata) == N_IMAGES_EXPECTED, (
        f"expected {N_IMAGES_EXPECTED} metadata rows, got {len(metadata)}"
    )
    assert metadata["split"].notna().all(), "some images got no split assigned"

    # 1) No lesion may appear in more than one split (the leakage guard).
    lesions_per_split = metadata.groupby("lesion_id")["split"].nunique()
    assert (lesions_per_split == 1).all(), "lesion_id crosses split boundaries!"

    # 2) Every class must be present in every split.
    class_table = metadata.groupby(["split", "dx"]).size().unstack(fill_value=0)
    assert (class_table > 0).all().all(), f"empty class in a split:\n{class_table}"

    # 3) All referenced image files must exist on disk.
    missing = [
        img_id
        for img_id in metadata["image_id"]
        if not (images_dir / f"{img_id}.jpg").exists()
    ]
    assert not missing, f"{len(missing)} images missing on disk, e.g. {missing[:5]}"

    print("[ok] all sanity checks passed")
    print("\nImages per split and class:")
    print(class_table)
    print("\nClass proportions per split (%):")
    print((class_table.div(class_table.sum(axis=1), axis=0) * 100).round(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_frac = args.train_frac if args.train_frac is not None else cfg["split"]["train_frac"]
    val_frac = args.val_frac if args.val_frac is not None else cfg["split"]["val_frac"]
    seed = args.seed if args.seed is not None else cfg["split"]["seed"]

    raw_dir = Path(cfg["data"]["raw_dir"])
    metadata_candidates = list(raw_dir.rglob("HAM10000_metadata*.csv"))

    extract_archives(raw_dir)
    metadata_candidates = metadata_candidates or list(raw_dir.rglob("HAM10000_metadata*.csv"))
    if not metadata_candidates:
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found under {raw_dir}. "
            "Download the dataset from Kaggle and place the zip (or its "
            "contents) in data/raw/ first."
        )

    images_dir = collect_images(raw_dir)
    metadata = pd.read_csv(metadata_candidates[0])
    metadata = build_splits(metadata, train_frac, val_frac, seed)
    sanity_checks(metadata, images_dir)

    out_csv = Path(cfg["data"]["splits_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv} ({len(metadata)} rows)")


if __name__ == "__main__":
    main()
