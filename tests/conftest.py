"""A tiny fake HAM10000 folder, so tests run without the real dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dermagan.labels import CLASSES


@pytest.fixture
def fake_ham(tmp_path: Path) -> Path:
    """10 lesions per class, 1-2 images per lesion, split over the two
    HAM10000 image folders like the real download."""
    rng = np.random.default_rng(0)
    root = tmp_path / "ham"
    parts = [root / "HAM10000_images_part_1", root / "HAM10000_images_part_2"]
    for p in parts:
        p.mkdir(parents=True)

    rows, n = [], 0
    for cls in CLASSES:
        for lesion in range(10):
            lesion_id = f"HAM_{cls}_{lesion:03d}"
            for _ in range(1 + lesion % 2):
                image_id = f"ISIC_{n:07d}"
                pixels = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(parts[n % 2] / f"{image_id}.jpg")
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": image_id,
                        "dx": cls,
                        "dx_type": "histo",
                        "age": 50,
                        "sex": "male",
                        "localization": "back",
                    }
                )
                n += 1
    pd.DataFrame(rows).to_csv(root / "HAM10000_metadata.csv", index=False)
    return root
