"""Shape and smoke tests for the PyTorch parts. Skipped if PyTorch is not
installed; CI installs the CPU build and runs them."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from dermagan.classifiers import build_model, remove_head, train_cnn  # noqa: E402
from dermagan.data import load_ham10000, split_by_lesion  # noqa: E402
from dermagan.gan import Discriminator, Generator, synthesize, train_gan  # noqa: E402
from dermagan.labels import CLASSES  # noqa: E402
from dermagan.torch_data import LesionDataset  # noqa: E402


@pytest.fixture
def splits(fake_ham, tmp_path):
    path = tmp_path / "splits.csv"
    split_by_lesion(load_ham10000(fake_ham)).to_csv(path, index=False)
    return path


def test_generator_and_discriminator_shapes():
    gen, disc = Generator(), Discriminator()
    labels = torch.tensor([0, 3, 6, 1])
    images = gen(torch.randn(4, 100), labels)
    assert images.shape == (4, 3, 64, 64)
    assert images.min() >= -1 and images.max() <= 1
    assert disc(images, labels).shape == (4,)


@pytest.mark.parametrize(
    "name,feat_dim", [("vgg16", 4096), ("googlenet", 1024), ("resnet50", 2048)]
)
def test_classifier_heads(name, feat_dim):
    model = build_model(name, pretrained=False, freeze_backbone=True).eval()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(("fc.", "classifier.6.")) for n in trainable)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert model(x).shape == (2, len(CLASSES))
        assert remove_head(model, name)(x).shape == (2, feat_dim)


def test_gan_train_and_synthesize(splits, tmp_path):
    ckpt = train_gan(splits, tmp_path / "gan", epochs=1, batch_size=8, num_workers=0, device="cpu")
    assert (tmp_path / "gan" / "samples" / "epoch_001.png").exists()
    written = synthesize(ckpt, tmp_path / "synthetic", {"df": 3, "vasc": 2, "nv": 0}, device="cpu")
    assert written == {"df": 3, "vasc": 2}
    assert len(list((tmp_path / "synthetic" / "df").glob("*.png"))) == 3

    train = LesionDataset(splits, "train", synthetic_dir=tmp_path / "synthetic")
    real = LesionDataset(splits, "train")
    assert len(train) == len(real) + 5
    with pytest.raises(ValueError):
        LesionDataset(splits, "test", synthetic_dir=tmp_path / "synthetic")


def test_train_cnn_smoke(splits, tmp_path):
    metrics = train_cnn(
        splits,
        "resnet50",
        tmp_path / "cnn",
        epochs=1,
        batch_size=8,
        image_size=64,
        pretrained=False,
        num_workers=0,
        device="cpu",
    )
    assert 0 <= metrics["accuracy"] <= 1
    for name in ("best.pt", "metrics.json", "history.csv", "confusion_matrix.png"):
        assert (tmp_path / "cnn" / name).exists()
    assert len(pd.read_csv(tmp_path / "cnn" / "history.csv")) == 1
