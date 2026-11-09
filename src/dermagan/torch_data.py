"""PyTorch datasets and transforms."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dermagan.data import IMAGE_EXTENSIONS
from dermagan.labels import CLASS_TO_ID

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classifier_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Transforms for the pretrained CNNs. Training adds the augmentations
    described in the thesis: rotation, shift, scale, flips and brightness."""
    steps: list = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def gan_transforms(image_size: int = 64) -> transforms.Compose:
    """GAN images are scaled to [-1, 1] to match the generator's tanh output."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def synthetic_records(synthetic_dir: str | Path) -> pd.DataFrame:
    """Read GAN images stored as <synthetic_dir>/<class code>/*.png."""
    rows = []
    for class_dir in sorted(Path(synthetic_dir).iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_ID:
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": str(path), "label": CLASS_TO_ID[class_dir.name]})
    return pd.DataFrame(rows, columns=["path", "label"])


class LesionDataset(Dataset):
    """Images listed in the splits CSV written by `dermagan prepare`.

    Synthetic GAN images can be added to the *training* split only; validation
    and test stay 100% real so the evaluation is not biased.
    """

    def __init__(
        self,
        splits_csv: str | Path,
        split: str,
        transform=None,
        synthetic_dir: str | Path | None = None,
    ):
        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split][["path", "label"]]
        if synthetic_dir is not None:
            if split != "train":
                raise ValueError("synthetic images may only be added to the train split")
            df = pd.concat([df, synthetic_records(synthetic_dir)], ignore_index=True)
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]
