import json

import numpy as np
import pytest

from dermagan.metrics import compute_metrics, plot_confusion_matrix, specificity
from dermagan.ml import train_and_evaluate
from dermagan.report import collect, to_markdown, write_report


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 0])
    prob = np.eye(7)[y]
    m = compute_metrics(y, y, prob)
    for key in (
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "specificity_mean",
        "roc_auc_weighted",
    ):
        assert m[key] == 1.0


def test_known_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = compute_metrics(y_true, y_pred, n_classes=2)
    assert m["accuracy"] == 0.75
    # class 0: precision 1, recall .5 ; class 1: precision 2/3, recall 1
    assert m["recall_weighted"] == 0.75
    assert m["precision_weighted"] == pytest.approx((1 + 2 / 3) / 2, abs=1e-4)
    # specificity: class 0 -> 2/2, class 1 -> 1/2
    assert specificity(np.array(y_true), np.array(y_pred), 2) == 0.75


CENTERS = np.random.default_rng(42).normal(0, 5, (7, 16))


def _fake_features(rng, n=140):
    """Seven well separated clusters; train and test share the same centers."""
    y = np.repeat(np.arange(7), n // 7)
    X = CENTERS[y] + rng.normal(0, 1, (len(y), CENTERS.shape[1]))
    return X, y


def test_classifiers_learn_separable_features():
    rng = np.random.default_rng(0)
    X, y = _fake_features(rng)
    Xt, yt = _fake_features(rng, n=70)
    results = train_and_evaluate({"X_train": X, "y_train": y, "X_test": Xt, "y_test": yt})
    assert set(results) == {"RandomForest", "KNN", "DecisionTree"}
    for m in results.values():
        assert m["accuracy"] > 0.8
        assert "roc_auc_weighted" in m


def test_classifiers_handle_class_missing_from_train():
    rng = np.random.default_rng(1)
    X, y = _fake_features(rng)
    keep = y != 6
    Xt, yt = _fake_features(rng, n=70)
    results = train_and_evaluate(
        {"X_train": X[keep], "y_train": y[keep], "X_test": Xt, "y_test": yt}
    )
    assert all(0 <= m["accuracy"] <= 1 for m in results.values())


def test_report(tmp_path):
    (tmp_path / "cnn" / "resnet50").mkdir(parents=True)
    (tmp_path / "cnn" / "resnet50" / "metrics.json").write_text(
        json.dumps({"accuracy": 0.8, "f1_weighted": 0.79})
    )
    (tmp_path / "ml" / "resnet50").mkdir(parents=True)
    (tmp_path / "ml" / "resnet50" / "metrics.json").write_text(
        json.dumps({"KNN": {"accuracy": 0.7}, "RandomForest": {"accuracy": 0.75}})
    )
    rows = collect(tmp_path)
    assert [r[0] for r in rows] == [
        "cnn/resnet50",
        "ml/resnet50 / KNN",
        "ml/resnet50 / RandomForest",
    ]
    md = to_markdown(rows)
    assert "| cnn/resnet50 | 0.800 |" in md
    write_report(tmp_path)
    assert (tmp_path / "results.md").exists()


def test_confusion_matrix_png(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix([0, 1, 2, 2], [0, 2, 2, 2], out, "test")
    assert out.stat().st_size > 0
