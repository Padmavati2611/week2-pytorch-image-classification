"""Command-line entry point for the Week 2 PyTorch image classification project.

Examples::

    python week2_pytorch_image_classification.py --train
    python week2_pytorch_image_classification.py --evaluate
    python week2_pytorch_image_classification.py --predict dataset/cats/image1.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.data_loader import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_SEED,
    DEFAULT_STD,
    DEFAULT_VAL_TEST_RATIO,
    SPLIT_MODES,
    DatasetError,
    create_dataloaders,
    seed_everything,
)
from src.evaluate import evaluate_model, load_history, save_all_plots
from src.model import describe_device, get_device
from src.predict import predict_path
from src.train import DEFAULT_MODEL_PATH, build_model_from_checkpoint, load_checkpoint, train_model


# --------------------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all configurable values."""
    parser = argparse.ArgumentParser(
        prog="week2_pytorch_image_classification.py",
        description="Train, evaluate and run a PyTorch CNN image classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    actions = parser.add_argument_group("actions")
    actions.add_argument("--train", action="store_true", help="Train the model on the dataset.")
    actions.add_argument("--evaluate", action="store_true", help="Evaluate the saved model on the test split.")
    actions.add_argument(
        "--predict",
        metavar="PATH",
        nargs="?",
        const="",
        default=None,
        help="Predict the class of an image (or every image in a folder).",
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--dataset", default="dataset", help="Dataset root folder (one sub-folder per class).")
    paths.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save / load the checkpoint.")
    paths.add_argument("--outputs-dir", default="outputs", help="Folder for plots and training history.")

    hyper = parser.add_argument_group("hyperparameters")
    hyper.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    hyper.add_argument("--batch-size", type=int, default=32, help="Samples per batch.")
    hyper.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    hyper.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Images are resized to this size.")
    hyper.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in the classifier head.")
    hyper.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of images used for training.")
    hyper.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of images used for validation.")
    hyper.add_argument("--split-mode", choices=list(SPLIT_MODES), default="auto",
                       help="auto: use train/val folders when present, else split randomly. "
                            "provided: require train/val folders. random: always split randomly.")
    hyper.add_argument("--val-test-ratio", type=float, default=DEFAULT_VAL_TEST_RATIO,
                       help="Fraction of a provided 'val' folder held out for testing when no "
                            "'test' folder exists.")
    hyper.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    hyper.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
    hyper.add_argument("--device", default="auto", help="Device: auto, cpu, cuda or mps.")

    predict_group = parser.add_argument_group("prediction")
    predict_group.add_argument("--top-k", type=int, default=3, help="How many class probabilities to display.")
    predict_group.add_argument("--show", action="store_true", help="Open a window showing the image (needs a GUI).")

    return parser


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def plot_training_curves(history: dict, outputs_dir: Path) -> None:
    """Draw the loss / accuracy curves produced by training."""
    from src.evaluate import plot_accuracy_curve, plot_loss_curve

    plot_loss_curve(history, outputs_dir / "training_loss.png")
    plot_accuracy_curve(history, outputs_dir / "accuracy_plot.png")


def command_train(args: argparse.Namespace) -> int:
    """Train the model, save the checkpoint and draw the training curves."""
    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Selected device: {describe_device(device)}")

    bundle = create_dataloaders(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        split_mode=args.split_mode,
        val_test_ratio=args.val_test_ratio,
    )
    print("\n".join(bundle.summary_lines()))

    result = train_model(
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
        image_size=bundle.image_size,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        model_path=args.model_path,
        outputs_dir=args.outputs_dir,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        seed=args.seed,
        split_source=bundle.split_source,
        split_mode=args.split_mode,
        val_test_ratio=args.val_test_ratio,
    )

    # Curves need the history, so they are drawn right after training.
    plot_training_curves(result["history"].to_dict(), Path(args.outputs_dir))
    print(f"Plots saved in   : {Path(args.outputs_dir).resolve()}")
    print("Run --evaluate next to score the test split and draw the confusion matrix.")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the saved checkpoint on the test split and plot results."""
    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Selected device: {describe_device(device)}")

    checkpoint = load_checkpoint(args.model_path, device=device)
    class_names: List[str] = list(checkpoint["class_names"])
    image_size: int = int(checkpoint["image_size"])
    if args.image_size != image_size:
        print(
            f"[i] --image-size {args.image_size} ignored: the checkpoint was trained at "
            f"{image_size}x{image_size}, which is also required for correct evaluation."
        )

    print("=" * 52)
    print("MODEL INFORMATION")
    print("=" * 52)
    print(f"Checkpoint      : {args.model_path}")
    print(f"Architecture    : {checkpoint.get('model_config', {}).get('architecture', 'unknown')}")
    print(f"Image size      : {image_size} x {image_size}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names     : {', '.join(class_names)}")
    print("=" * 52)

    # The image size and class names come from the checkpoint, so the dataset is
    # loaded with matching preprocessing and only the test split is needed.
    bundle = create_dataloaders(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        image_size=image_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        mean=tuple(checkpoint.get("norm_mean", DEFAULT_MEAN)),
        std=tuple(checkpoint.get("norm_std", DEFAULT_STD)),
        split_mode=args.split_mode,
        val_test_ratio=args.val_test_ratio,
    )

    missing = set(class_names) - set(bundle.class_names)
    if missing:
        raise DatasetError(
            "The dataset does not contain the classes the model was trained on: "
            f"{', '.join(sorted(missing))}.\nUse the same dataset folder you trained with."
        )

    model = build_model_from_checkpoint(checkpoint, device)
    report = evaluate_model(model, bundle.test_loader, device, class_names)

    history = load_history(checkpoint, outputs_dir=args.outputs_dir)
    paths = save_all_plots(report, history, outputs_dir=args.outputs_dir)
    print("\nVisualisations written:")
    for path in paths:
        print(f"  - {path}")

    return 0


def command_predict(args: argparse.Namespace) -> int:
    """Classify a new image (or a folder of images) with the saved model."""
    if args.predict == "":
        raise ValueError("--predict needs an image path, e.g. --predict dataset/cats/image1.jpg")

    device = get_device(args.device)
    target = Path(args.predict)
    if not target.exists():
        raise FileNotFoundError(
            f"No such file or folder: '{target}'\n"
            "Example: python week2_pytorch_image_classification.py --predict dataset/cats/image1.jpg"
        )

    results = predict_path(
        target=target,
        model_path=args.model_path,
        device=device,
        top_k=args.top_k,
    )

    for result in results:
        print(result.formatted())

    if args.show and len(results) == 1:
        from src.predict import show_image_with_prediction

        show_image_with_prediction(results[0])

    return 0


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, dispatch the requested command and handle errors."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.train or args.evaluate or args.predict is not None):
        parser.print_help()
        print("\n[!] Choose one action: --train, --evaluate or --predict PATH.")
        return 1

    try:
        if args.train:
            return command_train(args)
        if args.evaluate:
            return command_evaluate(args)
        return command_predict(args)
    except (DatasetError, FileNotFoundError, IsADirectoryError, ValueError) as error:
        print(f"\n[!] Error: {error}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[!] Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())