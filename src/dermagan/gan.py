"""Conditional DCGAN (cGAN) that generates 64x64 dermoscopic images for a
requested lesion class, used to augment the minority classes of HAM10000."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dermagan.labels import CLASSES, NUM_CLASSES
from dermagan.torch_data import LesionDataset, gan_transforms
from dermagan.utils import get_device, seed_everything

IMAGE_SIZE = 64


def _up(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _down(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """(noise z, class label) -> 3x64x64 image in [-1, 1]."""

    def __init__(
        self, nz: int = 100, n_classes: int = NUM_CLASSES, embed_dim: int = 50, ngf: int = 64
    ):
        super().__init__()
        self.nz, self.n_classes, self.embed_dim, self.ngf = nz, n_classes, embed_dim, ngf
        self.label_emb = nn.Embedding(n_classes, embed_dim)
        self.project = nn.Sequential(
            nn.Linear(nz + embed_dim, ngf * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            _up(ngf * 8, ngf * 4),  # 8x8
            _up(ngf * 4, ngf * 2),  # 16x16
            _up(ngf * 2, ngf),  # 32x32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),  # 64x64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        x = self.project(x).view(-1, self.ngf * 8, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    """(image, class label) -> real/fake logit. The label is embedded as an
    extra 64x64 input channel."""

    def __init__(self, n_classes: int = NUM_CLASSES, ndf: int = 64):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, IMAGE_SIZE * IMAGE_SIZE)
        self.net = nn.Sequential(
            _down(3 + 1, ndf, norm=False),  # 32x32
            _down(ndf, ndf * 2),  # 16x16
            _down(ndf * 2, ndf * 4),  # 8x8
            _down(ndf * 4, ndf * 8),  # 4x4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),  # 1x1
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_emb(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return self.net(torch.cat([images, label_map], dim=1)).view(-1)


def init_weights(module: nn.Module) -> None:
    """DCGAN initialisation (Radford et al., 2016)."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Sample every class equally often, so the GAN also learns rare classes."""
    counts = torch.bincount(torch.tensor(labels), minlength=NUM_CLASSES).float()
    weights = 1.0 / counts.clamp(min=1)
    sample_weights = weights[torch.tensor(labels)]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def save_generator(gen: Generator, path: Path) -> None:
    torch.save(
        {
            "state_dict": gen.state_dict(),
            "nz": gen.nz,
            "n_classes": gen.n_classes,
            "embed_dim": gen.embed_dim,
            "ngf": gen.ngf,
            "classes": list(CLASSES),
        },
        path,
    )


def load_generator(path: str | Path, device: torch.device) -> Generator:
    ckpt = torch.load(path, map_location=device)
    gen = Generator(ckpt["nz"], ckpt["n_classes"], ckpt["embed_dim"], ckpt["ngf"])
    gen.load_state_dict(ckpt["state_dict"])
    return gen.to(device).eval()


def train_gan(
    splits_csv: str | Path,
    out_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 2e-4,
    nz: int = 100,
    sample_every: int = 5,
    num_workers: int = 2,
    device: str | None = None,
    seed: int = 42,
) -> Path:
    """Train the cGAN on the training split and return the generator path."""
    from torchvision.utils import save_image

    seed_everything(seed)
    dev = get_device(device)
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    dataset = LesionDataset(splits_csv, "train", gan_transforms(IMAGE_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=balanced_sampler(dataset.labels),
        num_workers=num_workers,
        drop_last=True,
    )

    gen = Generator(nz=nz).to(dev)
    disc = Discriminator().to(dev)
    gen.apply(init_weights)
    disc.apply(init_weights)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    # Fixed noise: one row of 8 samples per class, to watch progress.
    fixed_labels = torch.arange(NUM_CLASSES, device=dev).repeat_interleave(8)
    fixed_z = torch.randn(len(fixed_labels), nz, device=dev)

    log_path = out_dir / "losses.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss_d", "loss_g"])

    print(f"training cGAN on {len(dataset)} images, device={dev}")
    for epoch in range(1, epochs + 1):
        gen.train()
        sum_d = sum_g = 0.0
        for real, labels in loader:
            real, labels = real.to(dev), labels.to(dev)
            n = real.size(0)
            z = torch.randn(n, nz, device=dev)
            fake = gen(z, labels)

            # Discriminator: real -> 0.9 (one-sided label smoothing), fake -> 0.
            opt_d.zero_grad()
            loss_d = bce(disc(real, labels), torch.full((n,), 0.9, device=dev)) + bce(
                disc(fake.detach(), labels), torch.zeros(n, device=dev)
            )
            loss_d.backward()
            opt_d.step()

            # Generator: make the discriminator call fakes real.
            opt_g.zero_grad()
            loss_g = bce(disc(fake, labels), torch.ones(n, device=dev))
            loss_g.backward()
            opt_g.step()

            sum_d += loss_d.item()
            sum_g += loss_g.item()

        steps = max(len(loader), 1)
        print(f"epoch {epoch:3d}/{epochs}  loss_D={sum_d / steps:.4f}  loss_G={sum_g / steps:.4f}")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, round(sum_d / steps, 5), round(sum_g / steps, 5)])

        if epoch % sample_every == 0 or epoch == epochs:
            gen.eval()
            with torch.no_grad():
                grid = gen(fixed_z, fixed_labels) * 0.5 + 0.5
            save_image(grid, out_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=8)
            save_generator(gen, out_dir / "generator.pt")

    return out_dir / "generator.pt"


@torch.no_grad()
def synthesize(
    checkpoint: str | Path,
    out_dir: str | Path,
    counts: dict[str, int],
    batch_size: int = 256,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Generate `counts[class]` images per class into <out_dir>/<class>/."""
    from torchvision.utils import save_image

    seed_everything(seed)
    dev = get_device(device)
    gen = load_generator(checkpoint, dev)
    out_dir = Path(out_dir)
    written = {}
    for cls, n in counts.items():
        if n <= 0:
            continue
        class_dir = out_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        label = CLASSES.index(cls)
        done = 0
        while done < n:
            b = min(batch_size, n - done)
            z = torch.randn(b, gen.nz, device=dev)
            labels = torch.full((b,), label, dtype=torch.long, device=dev)
            images = gen(z, labels) * 0.5 + 0.5
            for i, img in enumerate(images):
                save_image(img, class_dir / f"{cls}_gan_{done + i:05d}.png")
            done += b
        written[cls] = n
    return written
