import pandas as pd
import pytest

from dermagan.data import counts_to_balance, load_ham10000, split_by_lesion
from dermagan.labels import CLASS_TO_ID, CLASSES


def test_load_merges_both_image_folders(fake_ham):
    df = load_ham10000(fake_ham)
    meta = pd.read_csv(fake_ham / "HAM10000_metadata.csv")
    assert len(df) == len(meta)
    assert {"part_1", "part_2"} <= {p.split("images_")[-1].split("/")[0] for p in df["path"]}
    assert (df["label"] == df["dx"].map(CLASS_TO_ID)).all()


def test_missing_images_are_skipped(fake_ham):
    first = next((fake_ham / "HAM10000_images_part_1").glob("*.jpg"))
    first.unlink()
    df = load_ham10000(fake_ham)
    assert first.stem not in set(df["image_id"])


def test_unknown_diagnosis_raises(fake_ham):
    meta_path = fake_ham / "HAM10000_metadata.csv"
    meta = pd.read_csv(meta_path)
    meta.loc[0, "dx"] = "xyz"
    meta.to_csv(meta_path, index=False)
    with pytest.raises(ValueError, match="unknown diagnosis"):
        load_ham10000(fake_ham)


def test_split_keeps_lesions_together(fake_ham):
    df = split_by_lesion(load_ham10000(fake_ham))
    assert set(df["split"]) == {"train", "val", "test"}
    # Every lesion appears in exactly one split: no leakage of near duplicates.
    assert (df.groupby("lesion_id")["split"].nunique() == 1).all()


def test_split_proportions_and_stratification(fake_ham):
    df = split_by_lesion(load_ham10000(fake_ham))
    share = df["split"].value_counts(normalize=True)
    assert share["test"] == pytest.approx(0.2, abs=0.06)
    assert share["val"] == pytest.approx(0.16, abs=0.06)
    # Every class is present in every split.
    assert (df.groupby("split")["dx"].nunique() == len(CLASSES)).all()


def test_split_is_reproducible(fake_ham):
    a = split_by_lesion(load_ham10000(fake_ham), seed=1)
    b = split_by_lesion(load_ham10000(fake_ham), seed=1)
    assert a["split"].tolist() == b["split"].tolist()


def test_invalid_fraction():
    df = pd.DataFrame({"label": [0, 1] * 10, "lesion_id": range(20)})
    with pytest.raises(ValueError):
        split_by_lesion(df, test_size=0.3)


def test_counts_to_balance():
    df = pd.DataFrame({"dx": ["nv"] * 5 + ["mel"] * 2 + ["df"], "split": ["train"] * 8})
    counts = counts_to_balance(df, target=4)
    assert counts["nv"] == 0
    assert counts["mel"] == 2
    assert counts["df"] == 3
    assert counts["vasc"] == 4
    assert set(counts) == set(CLASSES)
