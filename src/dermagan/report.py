"""Collect every metrics.json under an outputs folder into one Markdown table."""

from __future__ import annotations

import json
from pathlib import Path

COLUMNS = [
    ("accuracy", "Accuracy"),
    ("precision_weighted", "Precision (w)"),
    ("recall_weighted", "Recall (w)"),
    ("f1_weighted", "F1 (w)"),
    ("f1_macro", "F1 (macro)"),
    ("specificity_mean", "Specificity"),
    ("roc_auc_weighted", "ROC AUC (w)"),
]


def collect(outputs_dir: str | Path) -> list[tuple[str, dict]]:
    """Return (run name, metrics) rows, one per model run."""
    outputs_dir = Path(outputs_dir)
    rows: list[tuple[str, dict]] = []
    for path in sorted(outputs_dir.rglob("metrics.json")):
        data = json.loads(path.read_text())
        run = path.parent.relative_to(outputs_dir).as_posix()
        if data and all(isinstance(v, dict) for v in data.values()):
            # Several classifiers in one file (classical ML on CNN features).
            rows.extend((f"{run} / {name}", m) for name, m in data.items())
        else:
            rows.append((run, data))
    return rows


def to_markdown(rows: list[tuple[str, dict]]) -> str:
    header = "| Run | " + " | ".join(title for _, title in COLUMNS) + " |"
    sep = "| --- |" + " ---: |" * len(COLUMNS)
    lines = [header, sep]
    for name, m in rows:
        cells = [f"{m[key]:.3f}" if key in m else "-" for key, _ in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(outputs_dir: str | Path, out_file: str | Path | None = None) -> str:
    rows = collect(outputs_dir)
    if not rows:
        raise FileNotFoundError(f"no metrics.json files under {outputs_dir}")
    md = "# Results (test set)\n\n" + to_markdown(rows)
    out_file = Path(out_file) if out_file else Path(outputs_dir) / "results.md"
    out_file.write_text(md)
    return md
