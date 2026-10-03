"""Week 2 - PyTorch Image Classification.

Public API used by ``week2_pytorch_image_classification.py``.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .data_loader import (  # noqa: F401
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_SEED,
    DEFAULT_STD,
    DEFAULT_VAL_TEST_RATIO,
    SPLIT_MODES,
    DataBundle,
    DatasetError,
    create_dataloaders,
    discover_provided_splits,
    get_transforms,
    seed_everything,
    validate_dataset_dir,
)
from .evaluate import (  # noqa: F401
    EvaluationReport,
    compute_metrics,
    evaluate_model,
    load_history,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    save_all_plots,
)
from .model import (  # noqa: F401
    ARCHITECTURE_NAME,
    ImageClassifier,
    count_parameters,
    create_model,
    describe_device,
    get_device,
)
from .predict import (  # noqa: F401
    PredictionResult,
    load_predictor,
    predict_image,
    predict_path,
)
from .train import (  # noqa: F401
    DEFAULT_MODEL_PATH,
    TrainHistory,
    build_model_from_checkpoint,
    load_checkpoint,
    train_model,
)

__all__ = [
    "ARCHITECTURE_NAME",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MEAN",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SEED",
    "DEFAULT_STD",
    "DEFAULT_VAL_TEST_RATIO",
    "DataBundle",
    "DatasetError",
    "EvaluationReport",
    "ImageClassifier",
    "PredictionResult",
    "SPLIT_MODES",
    "TrainHistory",
    "compute_metrics",
    "count_parameters",
    "create_dataloaders",
    "create_model",
    "describe_device",
    "discover_provided_splits",
    "evaluate_model",
    "get_device",
    "get_transforms",
    "load_checkpoint",
    "load_history",
    "load_predictor",
    "plot_accuracy_curve",
    "plot_confusion_matrix",
    "plot_loss_curve",
    "predict_image",
    "predict_path",
    "save_all_plots",
    "seed_everything",
    "train_model",
    "build_model_from_checkpoint",
]