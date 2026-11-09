"""End-to-end checks of the commands that do not need PyTorch."""

import json

import numpy as np
import pandas as pd

from dermagan.cli import main


def test_prepare_writes_splits(fake_ham, tmp_path, capsys):
    out = tmp_path / "splits.csv"
    main(["prepare", "--data-dir", str(fake_ham), "--out", str(out)])
    df = pd.read_csv(out)
    assert {"image_id", "lesion_id", "dx", "label", "path", "split"} <= set(df.columns)
    assert "train" in capsys.readouterr().out


def test_train_ml_and_report(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(7), 10)
    X = np.eye(7)[y] * 5 + rng.normal(0, 0.5, (70, 7))
    feats = tmp_path / "features" / "resnet50.npz"
    feats.parent.mkdir()
    np.savez(feats, X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y)

    outputs = tmp_path / "outputs"
    main(["train-ml", "--features", str(feats), "--out", str(outputs / "ml" / "resnet50")])
    metrics = json.loads((outputs / "ml" / "resnet50" / "metrics.json").read_text())
    assert metrics["RandomForest"]["accuracy"] > 0.9

    main(["report", "--outputs", str(outputs)])
    assert "RandomForest" in (outputs / "results.md").read_text()
