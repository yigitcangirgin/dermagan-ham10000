"""Dataset preparation for HAM10000 (no deep learning dependencies).

HAM10000 ships as ``HAM10000_metadata.csv`` plus images split across two
folders (``HAM10000_images_part_1`` and ``HAM10000_images_part_2``). This
module merges them into one table, attaches integer labels, and splits the
data into train / validation / test sets.

The split is done at **lesion level**: HAM10000 contains several images of
the same lesion, so a random per-image split would put near-duplicate images
in both train and test and inflate the scores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dermagan.labels import CLASS_TO_ID

METADATA_FILE = "HAM10000_metadata.csv"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_metadata(data_dir: Path) -> Path:
    """Return the path of ``HAM10000_metadata.csv`` anywhere under data_dir."""
    matches = sorted(data_dir.rglob(METADATA_FILE))
    if not matches:
        raise FileNotFoundError(f"{METADATA_FILE} not found under {data_dir}")
    return matches[0]


def index_images(data_dir: Path) -> dict[str, Path]:
    """Map image_id (file name without extension) to its path, across all
    sub folders. This merges the two HAM10000 image folders."""
    index: dict[str, Path] = {}
    for path in data_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            index.setdefault(path.stem, path)
    return index


def load_ham10000(data_dir: str | Path) -> pd.DataFrame:
    """Load metadata, attach image paths and integer labels.

    Returns a DataFrame with columns image_id, lesion_id, dx, label, path.
    Rows whose image file is missing are dropped (and reported).
    """
    data_dir = Path(data_dir)
    meta = pd.read_csv(find_metadata(data_dir))
    required = {"image_id", "lesion_id", "dx"}
    missing_cols = required - set(meta.columns)
    if missing_cols:
        raise ValueError(f"metadata is missing columns: {sorted(missing_cols)}")

    unknown = set(meta["dx"]) - set(CLASS_TO_ID)
    if unknown:
        raise ValueError(f"unknown diagnosis codes in metadata: {sorted(unknown)}")

    images = index_images(data_dir)
    meta["path"] = meta["image_id"].map(lambda i: str(images[i]) if i in images else None)
    n_missing = int(meta["path"].isna().sum())
    if n_missing:
        print(f"warning: {n_missing} images listed in metadata were not found; skipping them")
    meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    if meta.empty:
        raise FileNotFoundError(f"no HAM10000 images found under {data_dir}")

    meta["label"] = meta["dx"].map(CLASS_TO_ID).astype(int)
    return meta[["image_id", "lesion_id", "dx", "label", "path"]]


def _one_fold(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Hold out roughly `fraction` of df, stratified by label, grouped by lesion."""
    n_splits = round(1 / fraction)
    if n_splits < 2 or abs(n_splits * fraction - 1) > 1e-6:
        raise ValueError(f"fraction must be 1/k for an integer k >= 2, got {fraction}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, holdout_idx = next(splitter.split(df, df["label"], groups=df["lesion_id"]))
    return df.index[keep_idx], df.index[holdout_idx]


def split_by_lesion(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    """Add a ``split`` column (train / val / test).

    Mirrors the thesis protocol (80/20 train/test, then 80/20 train/val on the
    training part) but keeps all images of one lesion in the same split.
    """
    df = df.reset_index(drop=True).copy()
    trainval_idx, test_idx = _one_fold(df, test_size, seed)
    trainval = df.loc[trainval_idx]
    train_idx, val_idx = _one_fold(trainval, val_size, seed)

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def counts_to_balance(df: pd.DataFrame, target: int) -> dict[str, int]:
    """How many synthetic images each class needs so that its *training*
    count reaches `target`. Classes already above the target get 0."""
    train_counts = df[df["split"] == "train"]["dx"].value_counts()
    return {cls: max(0, target - int(train_counts.get(cls, 0))) for cls in CLASS_TO_ID}


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Image counts per class and split, for printing."""
    return df.pivot_table(
        index="dx", columns="split", values="image_id", aggfunc="count", fill_value=0
    )[["train", "val", "test"]]
