"""Classical classifiers (Random Forest, KNN, Decision Tree) trained on CNN
features, as in the thesis results table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dermagan.labels import NUM_CLASSES
from dermagan.metrics import compute_metrics


def build_classifiers(seed: int = 42) -> dict:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        # KNN is distance based, so features are standardised first.
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    """Load an .npz written by `dermagan extract-features`.

    Expected keys: X_train, y_train, X_test, y_test (X_val / y_val optional).
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


def train_and_evaluate(features: dict[str, np.ndarray], seed: int = 42) -> dict[str, dict]:
    """Fit each classifier on train (+ val) features and score it on test."""
    X_train, y_train = features["X_train"], features["y_train"]
    if "X_val" in features:
        # Classical models have no early stopping, so validation data is
        # simply added to the training set.
        X_train = np.concatenate([X_train, features["X_val"]])
        y_train = np.concatenate([y_train, features["y_val"]])

    results = {}
    for name, clf in build_classifiers(seed).items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(features["X_test"])
        proba = clf.predict_proba(features["X_test"])
        # predict_proba only has columns for classes seen in training.
        full = np.zeros((len(y_pred), NUM_CLASSES))
        full[:, clf.classes_] = proba
        results[name] = compute_metrics(features["y_test"], y_pred, full)
    return results
