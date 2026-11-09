"""Transfer learning with ImageNet-pretrained VGG16, GoogLeNet (Inception v1)
and ResNet50, plus CNN feature extraction for the classical classifiers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from dermagan.labels import NUM_CLASSES
from dermagan.metrics import compute_metrics, plot_confusion_matrix, save_metrics
from dermagan.torch_data import LesionDataset, classifier_transforms
from dermagan.utils import get_device, seed_everything

MODEL_NAMES = ("vgg16", "googlenet", "resnet50")


def _head_name(name: str) -> str:
    return "classifier.6" if name == "vgg16" else "fc"


def build_model(
    name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True
) -> nn.Module:
    """Load a pretrained backbone and replace its last layer with a new
    `num_classes` output layer."""
    if name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "googlenet":
        if pretrained:
            # With weights, torchvision drops the auxiliary heads by default.
            model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        else:
            # transform_input=True matches the pretrained variant, so a
            # fine-tuned checkpoint loads into this model with identical behaviour.
            model = models.googlenet(
                weights=None, aux_logits=False, init_weights=True, transform_input=True
            )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown model {name!r}, choose from {MODEL_NAMES}")

    if freeze_backbone:
        head = _head_name(name)
        for param_name, param in model.named_parameters():
            param.requires_grad = param_name.startswith(head + ".")
    return model


def remove_head(model: nn.Module, name: str) -> nn.Module:
    """Turn a classifier into a feature extractor (penultimate layer output)."""
    if name == "vgg16":
        model.classifier[6] = nn.Identity()
    else:
        model.fc = nn.Identity()
    return model


def _logits(output) -> torch.Tensor:
    # GoogLeNet returns a namedtuple when auxiliary heads are active.
    return output.logits if hasattr(output, "logits") else output


def _set_train_mode(model: nn.Module, freeze_backbone: bool) -> None:
    model.train()
    if freeze_backbone:
        # Keep the frozen backbone's BatchNorm statistics fixed.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred, y_prob) as numpy arrays."""
    model.eval()
    ys, probs = [], []
    for images, labels in loader:
        out = _logits(model(images.to(device)))
        probs.append(torch.softmax(out, dim=1).cpu())
        ys.append(labels)
    y_prob = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), y_prob.argmax(axis=1), y_prob


def _loaders(splits_csv, image_size, batch_size, num_workers, synthetic_dir=None):
    train_ds = LesionDataset(
        splits_csv,
        "train",
        classifier_transforms(image_size, train=True),
        synthetic_dir=synthetic_dir,
    )
    val_ds = LesionDataset(splits_csv, "val", classifier_transforms(image_size))
    test_ds = LesionDataset(splits_csv, "test", classifier_transforms(image_size))
    kw = {"batch_size": batch_size, "num_workers": num_workers}
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, **kw),
        DataLoader(test_ds, **kw),
    )


def train_cnn(
    splits_csv: str | Path,
    model_name: str,
    out_dir: str | Path,
    synthetic_dir: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    image_size: int = 224,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    num_workers: int = 2,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Train with Adam + cross-entropy, keep the checkpoint with the lowest
    validation loss, stop early after `patience` epochs without improvement,
    then evaluate the best model on the test set."""
    seed_everything(seed)
    dev = get_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl = _loaders(
        splits_csv, image_size, batch_size, num_workers, synthetic_dir
    )
    model = build_model(model_name, pretrained=pretrained, freeze_backbone=freeze_backbone).to(dev)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(
        f"{model_name}: train={len(train_dl.dataset)} val={len(val_dl.dataset)} "
        f"test={len(test_dl.dataset)} device={dev}"
    )
    best_val, bad_epochs, history = float("inf"), 0, []
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        _set_train_mode(model, freeze_backbone)
        train_loss = correct = seen = 0
        for images, labels in train_dl:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            out = _logits(model(images))
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        model.eval()
        val_loss = val_correct = val_seen = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(dev), labels.to(dev)
                out = _logits(model(images))
                val_loss += criterion(out, labels).item() * labels.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_seen += labels.size(0)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss / seen, 5),
            "train_acc": round(correct / seen, 5),
            "val_loss": round(val_loss / val_seen, 5),
            "val_acc": round(val_correct / val_seen, 5),
        }
        history.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()))

        if row["val_loss"] < best_val:
            best_val, bad_epochs = row["val_loss"], 0
            torch.save(
                {"model": model_name, "state_dict": model.state_dict(), "image_size": image_size},
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _plot_history(history, out_dir / "history.png", model_name)

    model.load_state_dict(torch.load(ckpt_path, map_location=dev)["state_dict"])
    y_true, y_pred, y_prob = predict(model, test_dl, dev)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    save_metrics(metrics, out_dir / "metrics.json")
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png", model_name)
    print(f"test: {metrics}")
    return metrics


def _plot_history(history: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_title("Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set_title("Accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def extract_features(
    splits_csv: str | Path,
    model_name: str,
    out_path: str | Path,
    checkpoint: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    device: str | None = None,
) -> Path:
    """Save penultimate-layer CNN features for train/val/test to an .npz.

    Uses the fine-tuned checkpoint when given, otherwise ImageNet weights.
    """
    dev = get_device(device)
    model = build_model(model_name, pretrained=checkpoint is None, freeze_backbone=False)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=dev)["state_dict"])
    model = remove_head(model, model_name).to(dev).eval()

    arrays = {}
    for split in ("train", "val", "test"):
        ds = LesionDataset(splits_csv, split, classifier_transforms(image_size))
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        feats, ys = [], []
        for images, labels in dl:
            feats.append(_logits(model(images.to(dev))).cpu().numpy())
            ys.append(labels.numpy())
        arrays[f"X_{split}"] = np.concatenate(feats)
        arrays[f"y_{split}"] = np.concatenate(ys)
        print(f"{split}: {arrays[f'X_{split}'].shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path
