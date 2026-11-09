"""Command line interface: `dermagan <command> [options]`.

PyTorch is imported only by the commands that need it, so `prepare`,
`train-ml` and `report` also work on machines without PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_prepare(args: argparse.Namespace) -> None:
    from dermagan.data import load_ham10000, split_by_lesion, split_summary

    df = split_by_lesion(load_ham10000(args.data_dir), args.test_size, args.val_size, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(split_summary(df).to_string())
    print(f"\nwrote {len(df)} rows to {out}")


def cmd_train_gan(args: argparse.Namespace) -> None:
    from dermagan.gan import train_gan

    path = train_gan(
        args.splits,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_every=args.sample_every,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
    )
    print(f"generator saved to {path}")


def cmd_synthesize(args: argparse.Namespace) -> None:
    import pandas as pd

    from dermagan.data import counts_to_balance
    from dermagan.gan import synthesize
    from dermagan.labels import CLASSES

    if args.per_class is not None:
        counts = {cls: args.per_class for cls in CLASSES}
    else:
        counts = counts_to_balance(pd.read_csv(args.splits), args.balance_to)
    written = synthesize(args.checkpoint, args.out, counts, device=args.device, seed=args.seed)
    print(f"generated: {json.dumps(written)}")


def cmd_train_cnn(args: argparse.Namespace) -> None:
    from dermagan.classifiers import train_cnn

    out = args.out or f"outputs/cnn/{args.model}{'+gan' if args.synthetic else ''}"
    train_cnn(
        args.splits,
        args.model,
        out,
        synthetic_dir=args.synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        image_size=args.image_size,
        freeze_backbone=not args.finetune,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
    )


def cmd_extract(args: argparse.Namespace) -> None:
    from dermagan.classifiers import extract_features

    out = args.out or f"outputs/features/{args.model}.npz"
    path = extract_features(
        args.splits,
        args.model,
        out,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        num_workers=args.workers,
        device=args.device,
    )
    print(f"features saved to {path}")


def cmd_train_ml(args: argparse.Namespace) -> None:
    from dermagan.metrics import save_metrics
    from dermagan.ml import load_features, train_and_evaluate

    results = train_and_evaluate(load_features(args.features), seed=args.seed)
    out = Path(args.out or f"outputs/ml/{Path(args.features).stem}") / "metrics.json"
    save_metrics(results, out)
    for name, m in results.items():
        print(f"{name:13s} {m}")
    print(f"saved to {out}")


def cmd_report(args: argparse.Namespace) -> None:
    from dermagan.report import write_report

    print(write_report(args.outputs, args.out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dermagan", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, torch_cmd: bool = False):
        sp.add_argument("--seed", type=int, default=42)
        if torch_cmd:
            sp.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
            sp.add_argument("--workers", type=int, default=2, help="DataLoader workers")

    sp = sub.add_parser("prepare", help="merge HAM10000 folders and make lesion-level splits")
    sp.add_argument("--data-dir", required=True, help="folder with HAM10000 csv and images")
    sp.add_argument("--out", default="data/splits.csv")
    sp.add_argument("--test-size", type=float, default=0.2)
    sp.add_argument("--val-size", type=float, default=0.2)
    common(sp)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train-gan", help="train the conditional DCGAN")
    sp.add_argument("--splits", default="data/splits.csv")
    sp.add_argument("--out", default="outputs/gan")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=2e-4)
    sp.add_argument("--sample-every", type=int, default=5)
    common(sp, torch_cmd=True)
    sp.set_defaults(func=cmd_train_gan)

    sp = sub.add_parser("synthesize", help="generate synthetic images with a trained generator")
    sp.add_argument("--checkpoint", default="outputs/gan/generator.pt")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--splits", default="data/splits.csv")
    group = sp.add_mutually_exclusive_group()
    group.add_argument(
        "--balance-to",
        type=int,
        default=1000,
        help="top up every class's training count to this number (default)",
    )
    group.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="generate exactly this many images for every class",
    )
    common(sp, torch_cmd=True)
    sp.set_defaults(func=cmd_synthesize)

    sp = sub.add_parser("train-cnn", help="transfer learning with a pretrained CNN")
    sp.add_argument("--model", choices=["vgg16", "googlenet", "resnet50"], required=True)
    sp.add_argument("--splits", default="data/splits.csv")
    sp.add_argument("--synthetic", default=None, help="folder of GAN images to add to train")
    sp.add_argument("--out", default=None)
    sp.add_argument("--epochs", type=int, default=20)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--patience", type=int, default=5)
    sp.add_argument("--image-size", type=int, default=224)
    sp.add_argument(
        "--finetune", action="store_true", help="train all layers instead of only the new head"
    )
    common(sp, torch_cmd=True)
    sp.set_defaults(func=cmd_train_cnn)

    sp = sub.add_parser("extract-features", help="save CNN features for train/val/test")
    sp.add_argument("--model", choices=["vgg16", "googlenet", "resnet50"], required=True)
    sp.add_argument("--splits", default="data/splits.csv")
    sp.add_argument("--checkpoint", default=None, help="fine-tuned best.pt (default: ImageNet)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--image-size", type=int, default=224)
    common(sp, torch_cmd=True)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("train-ml", help="Random Forest / KNN / Decision Tree on CNN features")
    sp.add_argument("--features", required=True, help=".npz from extract-features")
    sp.add_argument("--out", default=None)
    common(sp)
    sp.set_defaults(func=cmd_train_ml)

    sp = sub.add_parser("report", help="collect all metrics.json into a Markdown table")
    sp.add_argument("--outputs", default="outputs")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
