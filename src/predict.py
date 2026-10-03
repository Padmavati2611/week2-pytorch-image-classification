"""Inference helpers: classify new images with a saved checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .train import build_model_from_checkpoint, load_checkpoint

DEFAULT_MODEL_PATH = Path("models") / "image_classifier.pth"
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


@dataclass
class PredictionResult:
    """Result of classifying a single image."""

    image_path: Path
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    top_k: List[Tuple[str, float]]

    def formatted(self) -> str:
        lines = [
            "=" * 52,
            "PREDICTION",
            "=" * 52,
            f"Image           : {self.image_path}",
            f"Predicted Class : {self.predicted_class}",
            f"Confidence      : {self.confidence:.2f}%",
        ]
        if len(self.top_k) > 1:
            lines.append("-" * 52)
            lines.append("Top probabilities:")
            for name, probability in self.top_k:
                lines.append(f"  {name:<20s} {probability:6.2f}%")
        lines.append("=" * 52)
        return "\n".join(lines)


def build_predict_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Compose:
    """Deterministic preprocessing for single images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def load_predictor(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: torch.device | str = "cpu",
) -> Tuple[torch.nn.Module, List[str], transforms.Compose, int]:
    """Load checkpoint + rebuild model and preprocessing.

    Returns:
        ``(model, class_names, transform, image_size)``
    """
    device = torch.device(device)
    checkpoint = load_checkpoint(model_path, device=device)
    model = build_model_from_checkpoint(checkpoint, device)

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])
    mean = checkpoint.get("norm_mean", (0.485, 0.456, 0.406))
    std = checkpoint.get("norm_std", (0.229, 0.224, 0.225))

    return model, class_names, build_predict_transform(image_size, mean, std), image_size


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: Sequence[str],
    transform: transforms.Compose,
    device: torch.device | str = "cpu",
    top_k: int = 3,
) -> PredictionResult:
    """Classify a single image and return the predicted class + confidence."""
    device = torch.device(device)
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: '{image_path}'.")
    if image_path.is_dir():
        raise IsADirectoryError(
            f"'{image_path}' is a folder. Pass a single image file, or use --predict on a folder to batch classify."
        )
    if not image_path.is_file():
        raise ValueError(f"'{image_path}' is not a regular file.")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")  # handle greyscale / RGBA inputs
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image '{image_path}': unsupported file format.") from exc

    model.eval()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    k = max(1, min(top_k, len(class_names)))
    top_indices = probabilities.topk(k).indices.tolist()
    top_k_pairs = [(class_names[i], float(probabilities[i]) * 100.0) for i in top_indices]
    best_index = int(probabilities.argmax())

    return PredictionResult(
        image_path=image_path,
        predicted_class=class_names[best_index],
        confidence=float(probabilities[best_index]) * 100.0,
        probabilities={name: float(p) * 100.0 for name, p in zip(class_names, probabilities.tolist())},
        top_k=top_k_pairs,
    )


def collect_images(target: str | Path) -> List[Path]:
    """Resolve a path into a list of image files (single file or folder)."""
    target = Path(target)
    if target.is_dir():
        images = sorted(
            path
            for path in target.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(
                f"No images found in folder '{target}' (supported: {', '.join(IMAGE_EXTENSIONS)})."
            )
        return images
    return [target]


def predict_path(
    target: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: torch.device | str = "cpu",
    top_k: int = 3,
) -> List[PredictionResult]:
    """Predict for a single image or every image inside a folder."""
    device = torch.device(device)
    model, class_names, transform, _ = load_predictor(model_path, device=device)
    return [
        predict_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            transform=transform,
            device=device,
            top_k=top_k,
        )
        for image_path in collect_images(target)
    ]


def show_image_with_prediction(result: PredictionResult) -> None:
    """Display the image with the predicted label (needs a GUI backend)."""
    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        print("[!] Cannot display the image in this terminal - skipping preview.")
        return

    import matplotlib.pyplot as plt

    with Image.open(result.image_path) as image:
        figure, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image.convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{result.predicted_class} ({result.confidence:.2f}%)")
    plt.show()