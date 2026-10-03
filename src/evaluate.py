"""Evaluation: classification metrics, classification report and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

# Non-interactive backend so the script also works on headless machines / CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .model import describe_device

DEFAULT_OUTPUTS_DIR = Path("outputs")


@dataclass
class EvaluationReport:
    """Container for the metrics computed on the test split."""

    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    num_test_samples: int
    report_text: str
    predictions: List[int]
    targets: List[int]

    def summary_lines(self) -> List[str]:
        return [
            "=" * 52,
            "EVALUATION REPORT (test split)",
            "=" * 52,
            f"Test accuracy : {self.test_accuracy:.2f}%",
            f"Precision     : {self.precision:.4f}",
            f"Recall        : {self.recall:.4f}",
            f"F1-score      : {self.f1_score:.4f}",
            f"Test samples  : {self.num_test_samples}",
            "-" * 52,
            "Classification report (macro averaged over classes):",
            self.report_text,
            "=" * 52,
        ]


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[List[int], List[int]]:
    """Run the model over a loader and return ``(predictions, targets)``."""
    model.eval()
    all_predictions: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    return all_predictions, all_targets


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> EvaluationReport:
    """Compute accuracy, precision, recall, F1 and the confusion matrix."""
    labels = list(range(len(class_names)))

    accuracy = 100.0 * accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        labels=labels,
        digits=4,
        zero_division=0,
    )

    return EvaluationReport(
        test_accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        confusion_matrix=matrix,
        class_names=list(class_names),
        num_test_samples=len(y_true),
        report_text=report_text,
        predictions=list(y_pred),
        targets=list(y_true),
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> EvaluationReport:
    """Evaluate a trained model and print a professional report."""
    device = torch.device(device)
    print(f"Evaluating on device: {describe_device(device)}")
    predictions, targets = collect_predictions(model, loader, device)
    if not predictions:
        raise ValueError(
            "The test split is empty, so evaluation is not possible. "
            "Add more images to the dataset or adjust the split ratios."
        )

    report = compute_metrics(targets, predictions, class_names)
    for line in report.summary_lines():
        print(line)
    return report


def _finalize_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return output_path


def plot_loss_curve(history: Dict[str, List[float]], output_path: Path) -> Path:
    """Training vs validation loss."""
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Training Loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2, label="Validation Loss")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (CrossEntropy)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return _finalize_figure(fig, output_path)


def plot_accuracy_curve(history: Dict[str, List[float]], output_path: Path) -> Path:
    """Training vs validation accuracy."""
    epochs = history.get("epochs", [])
    train_acc = history.get("train_accuracy", [])
    val_acc = history.get("val_accuracy", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, marker="o", linewidth=2, label="Training Accuracy")
    ax.plot(epochs, val_acc, marker="s", linewidth=2, label="Validation Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    return _finalize_figure(fig, output_path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    normalize: bool = False,
) -> Path:
    """Confusion matrix with readable labels."""
    size = max(6.0, min(16.0, 1.0 + 0.8 * len(class_names)))
    fig, ax = plt.subplots(figsize=(size, size))

    display = matrix.astype("float")
    if normalize:
        display = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    image = ax.imshow(display, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""))
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = display.max() / 2.0 if display.size else 0.5
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    return _finalize_figure(fig, output_path)


def save_all_plots(
    report: EvaluationReport,
    history: Dict[str, List[float]],
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> List[Path]:
    """Create the three required figures inside ``outputs_dir``."""
    outputs_dir = Path(outputs_dir)
    paths = [
        plot_loss_curve(history, outputs_dir / "training_loss.png"),
        plot_accuracy_curve(history, outputs_dir / "accuracy_plot.png"),
        plot_confusion_matrix(report.confusion_matrix, report.class_names, outputs_dir / "confusion_matrix.png"),
    ]
    return paths


def load_history(checkpoint: Dict[str, object], outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR) -> Dict[str, List[float]]:
    """Return the training history stored in the checkpoint (or on disk)."""
    history = checkpoint.get("history")
    if isinstance(history, dict) and history.get("epochs"):
        return history  # type: ignore[return-value]

    history_path = Path(outputs_dir) / "training_history.json"
    if history_path.is_file():
        import json

        with open(history_path, "r", encoding="utf-8") as handle:
            return json.load(handle)  # type: ignore[no-any-return]

    raise ValueError(
        "No training history available. Re-run training (the history is stored in "
        "the checkpoint file) to generate the loss / accuracy plots."
    )