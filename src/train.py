"""Training pipeline: optimisation loop, metrics tracking and model saving."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import ARCHITECTURE_NAME, count_parameters, create_model, describe_device

DEFAULT_MODEL_PATH = Path("models") / "image_classifier.pth"
CHECKPOINT_VERSION = 1


@dataclass
class TrainHistory:
    """Per-epoch metrics collected during training."""

    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(round(train_loss, 6))
        self.train_accuracy.append(round(train_accuracy, 6))
        self.val_loss.append(round(val_loss, 6))
        self.val_accuracy.append(round(val_accuracy, 6))
        self.learning_rates.append(learning_rate)

    def to_dict(self) -> Dict[str, List[float]]:
        return asdict(self)

    def best_epoch(self) -> Optional[int]:
        if not self.val_accuracy:
            return None
        return int(self.epochs[self.val_accuracy.index(max(self.val_accuracy))])


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch (``optimizer=None`` runs evaluation mode instead)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    class_names: List[str],
    image_size: int,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    dropout: float = 0.5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    outputs_dir: str | Path = "outputs",
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    seed: int = 42,
    split_source: str = "random",
    split_mode: str = "auto",
    val_test_ratio: float = 0.5,
) -> Dict[str, object]:
    """Train the CNN and save the best checkpoint (by validation accuracy).

    Returns:
        Dictionary with the best metrics, history and checkpoint path.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    device = torch.device(device)
    torch.manual_seed(seed)

    print("=" * 52)
    print("TRAINING CONFIGURATION")
    print("=" * 52)
    print(f"Device          : {describe_device(device)}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : {batch_size}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Optimizer       : Adam")
    print(f"Loss function   : CrossEntropyLoss")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Val batches     : {len(val_loader)}")
    print("=" * 52)

    model = create_model(num_classes=num_classes, dropout=dropout, image_size=image_size)
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainHistory()
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = run_one_epoch(model, val_loader, criterion, None, device)

        history.add_epoch(
            epoch, train_loss, train_accuracy, val_loss, val_accuracy, current_lr
        )

        print(f"Epoch {epoch}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            # Deep copy keeps the best weights without saving to disk every epoch.
            best_state_dict = copy.deepcopy(
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            )

        print(f"Time for epoch: {time.time() - epoch_start:.1f}s")
        print("-" * 52)

    total_time = time.time() - start_time
    if best_state_dict is None:  # pragma: no cover - only if epochs == 0
        raise RuntimeError("Training finished without a valid model state.")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": best_state_dict,
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": image_size,
        "norm_mean": list(mean),
        "norm_std": list(std),
        "model_config": {
            "architecture": ARCHITECTURE_NAME,
            "num_classes": num_classes,
            "image_size": image_size,
            "dropout": dropout,
        },
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "total_training_seconds": round(total_time, 2),
            "split_mode": split_mode,
            "split_source": split_source,
            "val_test_ratio": val_test_ratio,
        },
        "history": history.to_dict(),
    }
    torch.save(checkpoint, model_path)

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    history_path = outputs_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history.to_dict(), handle, indent=2)

    print("=" * 52)
    print("TRAINING COMPLETE")
    print("=" * 52)
    print(f"Best epoch              : {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Total training time     : {total_time:.1f}s")
    print(f"Model saved to          : {model_path}")
    print(f"History saved to        : {history_path}")
    print("=" * 52)

    return {
        "model": model,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "model_path": model_path,
        "total_training_seconds": total_time,
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, object]:
    """Load a saved checkpoint and rebuild the model from its stored config.

    Raises:
        FileNotFoundError: when no checkpoint exists.
        ValueError: when the checkpoint is missing required metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No saved model found at '{checkpoint_path}'.\n"
            "Train the model first:\n"
            "  python week2_pytorch_image_classification.py --train"
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    except Exception as exc:  # pragma: no cover - corrupted file
        raise ValueError(f"Could not read '{checkpoint_path}': {exc}") from exc

    required = ("state_dict", "class_names", "num_classes", "image_size")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is missing fields: {missing}.\n"
            "Retrain the model to regenerate it."
        )

    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Recreate the saved architecture and load its weights."""
    config = checkpoint.get("model_config", {})
    dropout = float(config.get("dropout", 0.5)) if isinstance(config, dict) else 0.5
    model = create_model(
        num_classes=int(checkpoint["num_classes"]),
        dropout=dropout,
        image_size=int(checkpoint["image_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model