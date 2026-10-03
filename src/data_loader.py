"""Dataset loading, splitting and preprocessing for image classification.

Two layouts are supported.

**1. Flat dataset** - one folder per class; the split is created automatically::

    dataset/
    |-- class_a/
    |   |-- image1.jpg
    |   `-- image2.jpg
    `-- class_b/
        `-- image1.jpg

**2. Pre-split dataset** - the folders are supplied by the user::

    dataset/
    |-- train/
    |   |-- class_a/*.jpg
    |   `-- class_b/*.jpg
    |-- val/
    |   |-- class_a/*.jpg
    |   `-- class_b/*.jpg
    `-- test/            (optional)
        |-- class_a/*.jpg
        `-- class_b/*.jpg

With ``split_mode="auto"`` (the default) the pre-split layout is detected and used
as-is; otherwise the flat layout is split randomly. Class names are detected
automatically in both cases - nothing is hard-coded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Default preprocessing constants (ImageNet statistics, standard for transfer-style CNNs).
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_SEED: int = 42
DEFAULT_TRAIN_RATIO: float = 0.7
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15
#: Fraction of a provided ``val`` folder used as the test split when no ``test`` folder exists.
DEFAULT_VAL_TEST_RATIO: float = 0.5

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
)

#: Folder names recognised as pre-split directories (first match wins).
SPLIT_FOLDER_NAMES: Dict[str, Tuple[str, ...]] = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}

#: Accepted values for ``split_mode``.
SPLIT_MODES: Tuple[str, ...] = ("auto", "random", "provided")

#: Minimum number of images required before an automatic split is attempted.
MIN_IMAGES_FOR_SPLIT: int = 3


class DatasetError(RuntimeError):
    """Raised when the dataset directory is missing, empty or unusable."""


@dataclass
class DataBundle:
    """Everything the training / evaluation code needs about the data."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]
    class_to_idx: Dict[str, int]
    num_classes: int
    image_size: int
    dataset_dir: Path
    counts: Dict[str, int] = field(default_factory=dict)
    split_source: str = "random"

    @property
    def num_train(self) -> int:
        return self.counts.get("train", 0)

    @property
    def num_val(self) -> int:
        return self.counts.get("validation", 0)

    @property
    def num_test(self) -> int:
        return self.counts.get("test", 0)

    @property
    def total_images(self) -> int:
        return self.num_train + self.num_val + self.num_test

    def summary_lines(self) -> List[str]:
        """Human readable summary lines used by the CLI."""
        per_class = self.counts.get("per_class", {})
        lines = [
            "=" * 52,
            "DATASET SUMMARY",
            "=" * 52,
            f"Dataset path    : {self.dataset_dir}",
            f"Total images    : {self.total_images}",
            f"Training images : {self.num_train}",
            f"Validation imgs : {self.num_val}",
            f"Test images     : {self.num_test}",
            f"Number of classes: {self.num_classes}",
            f"Image size      : {self.image_size} x {self.image_size}",
            f"Class names     : {', '.join(self.class_names)}",
            f"Split source    : {self.split_source}",
        ]
        if per_class:
            lines.append("-" * 52)
            lines.append("Images per class:")
            for name in self.class_names:
                lines.append(f"  - {name}: {per_class.get(name, 0)}")
        lines.append("=" * 52)
        return lines


def find_image_files(directory: str | Path) -> List[Path]:
    """Return every image file inside ``directory`` (non-recursive)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )


def is_supported_image(file_path: str | Path) -> bool:
    """True when the file has a supported image extension.

    Passed to ``ImageFolder`` so stray files (``.txt``, ``.DS_Store``, ...) inside a
    class folder are ignored instead of crashing the data loader.
    """
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def discover_class_folders(dataset_dir: str | Path) -> List[str]:
    """List class folder names, ignoring hidden / private folders."""
    dataset_dir = Path(dataset_dir)
    return sorted(
        entry.name
        for entry in dataset_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith((".", "_"))
    )


def validate_dataset_dir(dataset_dir: str | Path) -> Tuple[List[str], Dict[str, int]]:
    """Validate the dataset folder.

    Returns ``(class_names, images_per_class)``.

    Raises:
        DatasetError: with an actionable message when something is missing.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise DatasetError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Create it and place one sub-folder per class inside, e.g.\n"
            "  dataset/cats/*.jpg\n"
            "  dataset/dogs/*.jpg\n"
            "See dataset/README.md for the expected format."
        )
    if not dataset_dir.is_dir():
        raise DatasetError(f"Dataset path is not a folder: {dataset_dir}")

    class_names = discover_class_folders(dataset_dir)
    if not class_names:
        raise DatasetError(
            f"No class sub-folders found inside '{dataset_dir}'.\n"
            "Each class needs its own folder, for example:\n"
            "  dataset/\n    class_1/\n      image1.jpg\n    class_2/\n      image1.jpg"
        )

    per_class: Dict[str, int] = {}
    empty_classes: List[str] = []
    for name in class_names:
        count = len(find_image_files(dataset_dir / name))
        per_class[name] = count
        if count == 0:
            empty_classes.append(name)

    if empty_classes:
        listing = "\n".join(f"  - {name}" for name in empty_classes)
        raise DatasetError(
            "These class folders contain no supported images "
            f"({', '.join(IMAGE_EXTENSIONS)}):\n{listing}\n"
            "Add images or remove the empty folders before training."
        )

    total = sum(per_class.values())
    if total < MIN_IMAGES_FOR_SPLIT:
        raise DatasetError(
            f"Only {total} image(s) found - at least {MIN_IMAGES_FOR_SPLIT} are needed "
            "to build train/validation/test splits."
        )
    return class_names, per_class


def get_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, eval_transform)``.

    Training uses augmentation for better generalisation; validation/test use a
    deterministic pipeline.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def discover_provided_splits(dataset_dir: str | Path) -> Optional[Dict[str, Path]]:
    """Detect a pre-split layout (``dataset/train/...``, ``dataset/val/...``).

    Returns a mapping ``{"train": Path, "val": Path, "test": Path}`` with only the
    splits that exist, or ``None`` when the layout is the flat one.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return None

    found: Dict[str, Path] = {}
    for split, aliases in SPLIT_FOLDER_NAMES.items():
        for alias in aliases:
            candidate = dataset_dir / alias
            # A valid split folder contains at least one class folder with images.
            if candidate.is_dir() and any(find_image_files(candidate / name) for name in discover_class_folders(candidate)):
                found[split] = candidate
                break

    return found if "train" in found and ("val" in found or "test" in found) else None


def validate_provided_splits(splits: Dict[str, Path]) -> List[str]:
    """Validate pre-split folders and confirm they share the same class folders.

    Returns:
        The class names, taken from the ``train`` split.

    Raises:
        DatasetError: when a split is empty or the class folders disagree.
    """
    reference: Optional[List[str]] = None
    for split, path in splits.items():
        try:
            class_names, per_class = validate_dataset_dir(path)
        except DatasetError as exc:
            raise DatasetError(f"Problem in the '{split}' split ({path}):\n{exc}") from exc

        if reference is None:
            reference = class_names
        elif set(class_names) != set(reference):
            only_here = sorted(set(class_names) - set(reference))
            missing = sorted(set(reference) - set(class_names))
            raise DatasetError(
                f"The '{split}' split has different classes than 'train'.\n"
                f"  only in '{split}': {', '.join(only_here) or '-'}\n"
                f"  missing from '{split}': {', '.join(missing) or '-'}\n"
                "Every split must contain exactly the same class folders."
            )
        print(f"[i] Provided '{split}' split: {sum(per_class.values())} images, classes {class_names}")

    if reference is None:
        raise DatasetError("No usable split folders were found.")
    return reference


def split_indices(
    num_samples: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministically split indices into train / validation / test sets."""
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio} + {val_ratio}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled = torch.randperm(num_samples, generator=generator).tolist()

    num_train = max(int(num_samples * train_ratio), 1)
    num_val = max(int(num_samples * val_ratio), 1)
    num_test = max(num_samples - num_train - num_val, 1)

    # Tiny datasets: rebalance so that no split ends up empty.
    if num_train + num_val + num_test > num_samples:
        num_test = max(num_samples - num_train - num_val, 1)
        num_train = max(num_samples - num_val - num_test, 1)

    train_idx = shuffled[:num_train]
    val_idx = shuffled[num_train : num_train + num_val]
    test_idx = shuffled[num_train + num_val : num_train + num_val + num_test]
    return train_idx, val_idx, test_idx


def _image_folder(path: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    return datasets.ImageFolder(str(path), transform=transform, is_valid_file=is_supported_image)


def _build_random_datasets(
    dataset_dir: Path,
    image_size: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Dataset, Dataset, Dataset, List[str], Dict[str, int]]:
    """Flat layout: split one ``ImageFolder`` into train / validation / test subsets."""
    class_names, per_class = validate_dataset_dir(dataset_dir)
    train_transform, eval_transform = get_transforms(image_size, mean, std)

    train_dataset = _image_folder(dataset_dir, train_transform)
    eval_dataset = _image_folder(dataset_dir, eval_transform)

    total = len(train_dataset)
    train_idx, val_idx, test_idx = split_indices(total, train_ratio, val_ratio, seed)

    return (
        Subset(train_dataset, train_idx),
        Subset(eval_dataset, val_idx),
        Subset(eval_dataset, test_idx),
        class_names,
        per_class,
    )


def _build_provided_datasets(
    dataset_dir: Path,
    image_size: int,
    seed: int,
    val_test_ratio: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Dataset, Dataset, Dataset, List[str], Dict[str, int]]:
    """Pre-split layout: use the user's ``train`` / ``val`` / ``test`` folders as-is.

    When no ``test`` folder exists, a held-out test split is carved out of ``val``
    using ``val_test_ratio`` so the model can still be evaluated honestly.
    """
    splits = discover_provided_splits(dataset_dir)
    if splits is None:
        raise DatasetError(
            f"No pre-split folders found in '{dataset_dir}'.\n"
            "Expected 'train' and ('val' or 'test') sub-folders, each containing "
            "one folder per class."
        )
    class_names = validate_provided_splits(splits)

    train_transform, eval_transform = get_transforms(image_size, mean, std)
    train_dataset = _image_folder(splits["train"], train_transform)

    if "test" in splits:
        val_dataset = _image_folder(splits["val"], eval_transform) if "val" in splits else None
        test_dataset = _image_folder(splits["test"], eval_transform)
        if val_dataset is None:
            # Only train + test were provided: hold out part of train for validation.
            train_idx, val_idx = _holdout_from_train(train_dataset, seed)
            train_dataset = Subset(train_dataset, train_idx)
            val_dataset = Subset(
                _image_folder(splits["train"], eval_transform), val_idx
            )
    else:
        val_source = _image_folder(splits["val"], eval_transform)
        total = len(val_source)
        num_test = max(int(total * val_test_ratio), 1)
        num_val = max(total - num_test, 1)
        if num_val + num_test > total:  # tiny val folder: keep at least one of each
            num_val = max(total - 1, 1)
            num_test = max(total - num_val, 1)

        generator = torch.Generator()
        generator.manual_seed(seed)
        shuffled = torch.randperm(total, generator=generator).tolist()
        val_dataset = Subset(val_source, shuffled[:num_val])
        test_dataset = Subset(val_source, shuffled[num_val : num_val + num_test])
        print(
            f"[i] No 'test' folder found: {num_test} of the {total} 'val' images are held out "
            f"for testing (val_test_ratio={val_test_ratio})."
        )

    per_class = {
        name: len(find_image_files(splits["train"] / name)) for name in class_names
    }
    return train_dataset, val_dataset, test_dataset, class_names, per_class


def _holdout_from_train(dataset: Dataset, seed: int) -> Tuple[List[int], List[int]]:
    """Split a training folder into (train, validation) indices for validation."""
    total = len(dataset)
    num_val = max(int(total * 0.15), 1)
    num_train = max(total - num_val, 1)
    if num_train + num_val > total:
        num_val = max(total - 1, 1)
        num_train = max(total - num_val, 1)

    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled = torch.randperm(total, generator=generator).tolist()
    return shuffled[:num_train], shuffled[num_train : num_train + num_val]


def create_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    num_workers: int = 0,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    pin_memory: bool | None = None,
    split_mode: str = "auto",
    val_test_ratio: float = DEFAULT_VAL_TEST_RATIO,
) -> DataBundle:
    """Build train / validation / test dataloaders.

    Args:
        dataset_dir: Root folder (class folders, or train/val/test folders).
        batch_size: Samples per batch.
        image_size: Spatial size images are resized to.
        seed: Seed used for the reproducible split.
        train_ratio / val_ratio: Fractions used when splitting a flat dataset.
        num_workers: DataLoader workers (0 is the safest default on Windows).
        mean / std: Normalisation statistics.
        pin_memory: Defaults to True when CUDA is available.
        split_mode: ``auto`` (use a pre-split layout when detected), ``random``
            (always split the flat dataset) or ``provided`` (require a pre-split layout).
        val_test_ratio: Fraction of a provided ``val`` folder held out as the test set
            when no ``test`` folder exists.

    Returns:
        DataBundle with dataloaders, class names and split sizes.
    """
    dataset_dir = Path(dataset_dir)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")

    provided_splits = discover_provided_splits(dataset_dir)
    use_provided = provided_splits is not None and split_mode in ("auto", "provided")

    if split_mode == "provided" and not use_provided:
        raise DatasetError(
            f"--split-mode provided was requested but '{dataset_dir}' has no 'train' and "
            "('val' or 'test') class folders. Use --split-mode auto or random instead."
        )

    if use_provided:
        train_dataset, val_dataset, test_dataset, class_names, per_class = _build_provided_datasets(
            dataset_dir, image_size, seed, val_test_ratio, mean, std
        )
        split_source = "provided (train/val folders used as-is)"
    else:
        train_dataset, val_dataset, test_dataset, class_names, per_class = _build_random_datasets(
            dataset_dir, image_size, seed, train_ratio, val_ratio, mean, std
        )
        split_source = "random (automatic train/val/test split)"

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    common: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        class_to_idx={name: idx for idx, name in enumerate(class_names)},
        num_classes=len(class_names),
        image_size=image_size,
        dataset_dir=dataset_dir,
        counts={
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
            "per_class": dict(per_class),
        },
        split_source=split_source,
    )


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Make Python, NumPy and PyTorch runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)