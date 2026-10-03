# Week 2 – PyTorch Image Classification

A complete, beginner-friendly but professionally structured **image classification system**
built with **PyTorch** and a **Convolutional Neural Network (CNN)**.

The project trains a CNN on your own image dataset, evaluates it on a held-out test split,
saves the best model, generates loss/accuracy/confusion-matrix figures, and predicts the class
of any new image from the command line.

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Objectives](#2-objectives)
3. [Features](#3-features)
4. [Technologies Used](#4-technologies-used)
5. [Project Structure](#5-project-structure)
6. [Dataset Format](#6-dataset-format)
7. [Installation](#7-installation)
8. [How to Train](#8-how-to-train)
9. [How to Evaluate](#9-how-to-evaluate)
10. [How to Make Predictions](#10-how-to-make-predictions)
11. [Example Output](#11-example-output)
12. [Results](#12-results)
13. [Configuration Reference](#13-configuration-reference)
14. [Model Architecture](#14-model-architecture)
15. [How the Code Works](#15-how-the-code-works)
16. [Troubleshooting](#16-troubleshooting)
17. [Future Improvements](#17-future-improvements)

---

## 1. Project Description

A self-contained image classification pipeline written in plain Python (no notebooks).
It uses `torchvision.datasets.ImageFolder`, so the dataset is organised as **one folder per
class** and the class names are detected automatically — nothing is hard-coded.

The same script handles the full workflow:

```text
dataset/  ->  train  ->  save best model  ->  evaluate  ->  plots  ->  predict new images
```

Everything is controlled by command-line flags, so you can change the dataset path,
image size, number of epochs, batch size or learning rate without editing the code.

## 2. Objectives

* Learn how a CNN image classifier is built in PyTorch, step by step.
* Understand dataset loading, augmentation, normalisation and a reproducible train/val/test split.
* Understand the training loop: forward pass, loss, backpropagation, optimiser step.
* Evaluate a model properly (accuracy, precision, recall, F1, confusion matrix).
* Produce publication-style plots of the training behaviour.
* Save and reuse a trained model for inference on new images.
* Produce clean, readable, well-structured code suitable for review.

## 3. Features

| Feature | Details |
|---|---|
| Automatic class detection | Class names come from the dataset folder names |
| Two dataset layouts | Flat class folders (auto 70/15/15 split) **or** your own `train`/`val`/`test` folders |
| Automatic splitting | Train / validation / test (70 / 15 / 15) with a fixed `--seed` |
| Augmentation | `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` |
| Preprocessing | Resize → ToTensor → Normalize (ImageNet statistics) |
| CNN architecture | 4 convolutional blocks with BatchNorm + ReLU + MaxPool + Dropout, then a dense head |
| Adaptive model | Automatically adjusts to **any** number of classes and **any** input image size |
| Hardware support | Uses CUDA automatically when available, otherwise CPU (also MPS on Apple) |
| Optimisation | `CrossEntropyLoss` + `Adam` (learning rate configurable) |
| Best-model saving | Keeps the checkpoint with the highest validation accuracy |
| Rich checkpoint | Stores weights, class names, image size, normalisation values and full history |
| Evaluation | Accuracy, macro precision / recall / F1, confusion matrix, `sklearn` classification report |
| Visualisation | Loss curve, accuracy curve and confusion matrix saved as PNG |
| Prediction | Softmax confidence for a single image **or** every image in a folder |
| Robust CLI | Clear, actionable error messages for missing dataset, model or image |
| Reproducible | Seeds Python, NumPy and PyTorch; `--seed` controls the split |

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Language |
| **PyTorch** | Tensors, `nn.Module`, autograd, training loop |
| **Torchvision** | `ImageFolder`, image transforms, augmentation |
| **NumPy** | Seeding and array helpers |
| **Matplotlib** | Loss / accuracy / confusion-matrix figures |
| **Pillow** | Reading image files |
| **scikit-learn** | `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, `classification_report` |

## 5. Project Structure

```text
Week 2 – PyTorch Image Classification/
│
├── dataset/                          # YOU provide the images
│   ├── README.md
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── val/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/                         (optional)
│       ├── cats/
│       └── dogs/
│
├── models/                           # trained weights
│   ├── .gitkeep
│   └── image_classifier.pth          # created by --train
│
├── outputs/                          # generated figures
│   ├── .gitkeep
│   ├── training_loss.png             # created by --train
│   ├── accuracy_plot.png             # created by --train
│   ├── confusion_matrix.png          # created by --evaluate
│   └── training_history.json         # created by --train
│
├── src/
│   ├── __init__.py                   # package exports
│   ├── data_loader.py                # dataset discovery, transforms, split, dataloaders
│   ├── model.py                      # CNN definition + device detection
│   ├── train.py                      # training loop + checkpoint save/load
│   ├── evaluate.py                   # metrics + plots
│   └── predict.py                    # inference helpers
│
├── week2_pytorch_image_classification.py   # CLI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Dataset Format

> **You must supply your own images.** The `dataset/` folder ships empty on purpose.

Two layouts are supported. Class names are read from the folder names, so nothing is
hard-coded.

### Layout A — pre-split dataset (used by this project)

The split is supplied by you and used **as-is**:

```text
dataset/
├── train/
│   ├── cats/        *.jpg
│   └── dogs/        *.jpg
├── val/
│   ├── cats/        *.jpg
│   └── dogs/        *.jpg
└── test/            (optional)
    ├── cats/        *.jpg
    └── dogs/        *.jpg
```

If there is no `test/` folder, `--val-test-ratio` (default `0.5`) of `val/` is held out
automatically so the model can still be evaluated on unseen data.

### Layout B — flat dataset

One folder per class; the project splits it automatically 70 / 15 / 15:

```text
dataset/
├── cats/
│   ├── image1.jpg
│   └── ...
├── dogs/
│   ├── image1.jpg
│   └── ...
└── horses/
    └── ...
```

A directory is only treated as pre-split when it contains `train` plus `val`/`test` folders
that themselves contain class folders — so a class literally named `train` still works.

### Rules

* **Folder name = class name**, kept exactly as written.
* **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`.
* **Minimum size:** 3 images in total.
  **Recommended:** 10+ images per class so validation and test scores are meaningful.
* **Images may have different sizes** — they are resized to `--image-size` (default `224`).
* **Grayscale and RGBA images** are converted to RGB automatically.
* **Stray non-image files** (`.txt`, `.DS_Store`, ...) inside a class folder are ignored.
* **Empty class folders raise an error** instead of being silently ignored.
* **Every split must contain the same class folders**, otherwise you get a clear error.
* **The automatic split is reproducible** — controlled by `--seed` (default `42`).

More details are in [`dataset/README.md`](dataset/README.md).

## 7. Installation

### Step 1 — get the code

```bash
git clone <your-repository-url>
cd "Week 2 – PyTorch Image Classification"
```

### Step 2 — create a virtual environment

**Windows (PowerShell / CMD):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

In **VS Code**, press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv\Scripts\python.exe`.

### Step 3 — install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU (optional).** To train on an NVIDIA GPU, install the CUDA build of PyTorch first
> from <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements.txt`.
> Skip this if you only have a CPU — the project runs fine on CPU.

### Step 4 — add your images

Copy your class folders into `dataset/` so that `dataset/<class_name>/*.jpg` is true.

### Step 5 — check the installation

```bash
python -c "import torch, torchvision, sklearn; print(torch.__version__)"
```

## 8. How to Train

```bash
python week2_pytorch_image_classification.py --train
```

With custom settings:

```bash
python week2_pytorch_image_classification.py --train --epochs 25 --batch-size 16 --lr 0.0005 --image-size 224
```

Force the pre-split layout (fail loudly if `train`/`val` folders are missing):

```bash
python week2_pytorch_image_classification.py --train --split-mode provided
```

Training prints the detected device, the dataset summary, a configuration block and one block
per epoch. It saves:

* the checkpoint with the **best validation accuracy** → `models/image_classifier.pth`
* the per-epoch history → `outputs/training_history.json`
* `outputs/training_loss.png` and `outputs/accuracy_plot.png`

> **Timing on CPU.** This dataset (1200 images) takes ~27 s per epoch at 64×64 and roughly
> 5–6 min per epoch at the 224×224 default. For quick experiments use `--image-size 96`; for a
> serious run use a GPU (`pip install torch` from the CUDA build on pytorch.org).

## 9. How to Evaluate

```bash
python week2_pytorch_image_classification.py --evaluate
```

This loads the saved checkpoint, rebuilds the model from its stored configuration, scores the
**test split** and writes `outputs/confusion_matrix.png` (the loss/accuracy curves are
refreshed from the stored history).

```
Test accuracy : xx.xx%
Precision     : x.xxxx      (macro average)
Recall        : x.xxxx      (macro average)
F1-score      : x.xxxx      (macro average)
```

followed by a full per-class `sklearn` classification report.

## 10. How to Make Predictions

**Single image:**

```bash
python week2_pytorch_image_classification.py --predict path/to/image.jpg
```

**Whole folder (every image inside is classified):**

```bash
python week2_pytorch_image_classification.py --predict dataset/cats
```

**Options:** `--top-k 5` to show more probabilities, `--show` to open an image preview window
(requires a graphical backend).

## 11. Example Output

> The output below comes from a **throwaway verification run on generated placeholder images**
> (flat colour squares, 3 classes, 2 epochs, 32×32 px) used only to confirm the code executes.
> For real numbers on the Dogs vs Cats data see [Results](#12-results).

Training:

```text
Selected device: CPU
====================================================
DATASET SUMMARY
====================================================
Dataset path    : dataset
Total images    : 30
Training images : 21
Validation imgs : 4
Test images     : 5
Number of classes: 3
Image size      : 32 x 32
Class names     : class_a, class_b, class_c
====================================================
Epoch 1/2
Training Loss: 1.0041
Training Accuracy: 66.67%
Validation Loss: 1.1110
Validation Accuracy: 50.00%
Time for epoch: 0.4s
----------------------------------------------------
Epoch 2/2
Training Loss: 0.5444
Training Accuracy: 90.48%
Validation Loss: 1.0326
Validation Accuracy: 50.00%
Time for epoch: 0.2s
----------------------------------------------------
====================================================
TRAINING COMPLETE
====================================================
Best epoch              : 1
Best validation accuracy: 50.00%
Model saved to          : models\image_classifier.pth
```

Evaluation:

```text
====================================================
EVALUATION REPORT (test split)
====================================================
Test accuracy : 20.00%
Precision     : 0.0667
Recall        : 0.3333
F1-score      : 0.1111
Test samples  : 5
----------------------------------------------------
Classification report (macro averaged over classes):
              precision    recall  f1-score   support

     class_a     0.0000    0.0000    0.0000         2
     class_b     0.2000    1.0000    0.3333         1
     class_c     0.0000    0.0000    0.0000         2

    accuracy                         0.2000         5
   macro avg     0.0667    0.3333    0.1111         5
weighted avg     0.0400    0.2000    0.0667         5
```

Prediction:

```text
====================================================
PREDICTION
====================================================
Image           : dataset/class_b/example.jpg
Predicted Class : class_b
Confidence      : 34.82%
----------------------------------------------------
Top probabilities:
  class_b               34.82%
  class_c               34.34%
  class_a               30.85%
====================================================
```

Error message example:

```text
[!] Error: No class sub-folders found inside 'dataset'.
Each class needs its own folder, for example:
  dataset/
    class_1/
      image1.jpg
    class_2/
      image1.jpg
```

## 12. Results

### Baseline run included in this repository

The `models/image_classifier.pth` and the three figures in `outputs/` come from a
**verification run** used to confirm the pipeline works on the real dataset. It is
deliberately short, so the numbers are **not** representative of what this model can reach:

```bash
python week2_pytorch_image_classification.py --train --epochs 1 --image-size 64 --batch-size 32
```

| Setting | Value |
|---|---|
| Dataset | Dogs vs Cats subset, `cats` / `dogs` |
| Train / Val / Test images | 1000 / 100 / 100 (50 % of `val/` held out) |
| Image size | 64 × 64 (**not** the 224 default) |
| Epochs | **1** |
| Device | CPU |
| Best validation accuracy | 52.00 % |
| **Test accuracy** | **55.00 %** |
| Precision (macro) | 0.5955 |
| Recall (macro) | 0.5564 |
| F1-score (macro) | 0.5022 |

**A single epoch at 64 px is barely above the 50 % chance level.** The purpose of this run is
to prove the code executes end to end, not to demonstrate accuracy. Retrain before drawing any
conclusion — see below.

<details>
<summary>Console output of the verification run (click to expand)</summary>

```text
Selected device: CPU
[i] Provided 'train' split: 1000 images, classes ['cats', 'dogs']
[i] Provided 'val' split: 200 images, classes ['cats', 'dogs']
[i] No 'test' folder found: 100 of the 200 'val' images are held out for testing (val_test_ratio=0.5).
====================================================
DATASET SUMMARY
====================================================
Dataset path    : dataset
Total images    : 1200
Training images : 1000
Validation imgs : 100
Test images     : 100
Number of classes: 2
Image size      : 64 x 64
Class names     : cats, dogs
Split source    : provided (train/val folders used as-is)
----------------------------------------------------
Images per class:
  - cats: 500
  - dogs: 500
====================================================
Epoch 1/1
Training Loss: 0.7049
Training Accuracy: 54.60%
Validation Loss: 0.7138
Validation Accuracy: 52.00%
Time for epoch: 27.3s
----------------------------------------------------

Test accuracy : 55.00%
Precision     : 0.5955
Recall        : 0.5564
F1-score      : 0.5022
```

</details>

### Reproducing a meaningful run

```bash
# CPU-friendly (~30 s/epoch)
python week2_pytorch_image_classification.py --train --epochs 15 --image-size 96 --batch-size 32

# standard configuration (use a GPU: ~12x slower per epoch on CPU)
python week2_pytorch_image_classification.py --train --epochs 20 --image-size 224 --batch-size 32

python week2_pytorch_image_classification.py --evaluate
```

Then replace the table above with your own numbers and drop your figures in:

```text
outputs/training_loss.png
outputs/accuracy_plot.png
outputs/confusion_matrix.png
```

<!-- RESULTS_PLACEHOLDER -->

### Honest expectations

Dogs vs cats is a relatively easy task — a well-trained CNN on ~1000 labelled images should
reach **85–95 %** test accuracy. The 55 % above is a training-budget artefact (1 epoch,
64 px inputs), not a limitation of the architecture. Report the number of images per class
alongside any accuracy you publish.

## 13. Configuration Reference

All values are optional flags — defaults are shown by `--help`.

| Flag | Default | Description |
|---|---|---|
| `--train` | – | Train the model |
| `--evaluate` | – | Evaluate the saved model on the test split |
| `--predict PATH` | – | Predict an image, or every image in a folder |
| `--dataset PATH` | `dataset` | Dataset root folder |
| `--model-path PATH` | `models/image_classifier.pth` | Checkpoint location |
| `--outputs-dir PATH` | `outputs` | Where figures are written |
| `--epochs N` | `10` | Training epochs |
| `--batch-size N` | `32` | Samples per batch |
| `--lr FLOAT` | `0.001` | Adam learning rate |
| `--image-size N` | `224` | Resized image size (square) |
| `--dropout FLOAT` | `0.5` | Dropout in the classifier head |
| `--train-ratio FLOAT` | `0.7` | Fraction used for training (flat layout only) |
| `--val-ratio FLOAT` | `0.15` | Fraction used for validation (flat layout only) |
| `--split-mode NAME` | `auto` | `auto` uses `train`/`val` folders when present, else splits randomly; `provided` requires them; `random` always splits randomly |
| `--val-test-ratio FLOAT` | `0.5` | Fraction of a provided `val` folder held out as the test set when no `test` folder exists |
| `--seed N` | `42` | Seed for the split and weight initialisation |
| `--num-workers N` | `0` | DataLoader workers (keep `0` on Windows) |
| `--device NAME` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `--top-k N` | `3` | Probabilities shown by `--predict` |
| `--show` | off | Open an image preview (needs a GUI backend) |

Typical time per epoch: roughly 10–40 s for 1 000 images at 224×224 on CPU, much faster on a
GPU. Lower `--image-size` to `64` or `32` for fast experiments on small datasets.

## 14. Model Architecture

`Week2CNN` (`src/model.py`) — a compact VGG-style CNN of about **1.2 M parameters**:

| Block | Layers | Output channels |
|---|---|---|
| Block 1 | Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool2 → Dropout2d(0.10) | 32 |
| Block 2 | Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool2 → Dropout2d(0.15) | 64 |
| Block 3 | Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool2 → Dropout2d(0.20) | 128 |
| Block 4 | Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool2 → Dropout2d(0.25) | 256 |
| Head | AdaptiveAvgPool → Dropout → Linear → ReLU → Dropout → **Linear(num_classes)** | — |

Design notes:

* **Batch normalisation** after every convolution stabilises and speeds up training.
* **`Dropout2d`** inside the blocks and **`Dropout`** in the head reduce overfitting.
* **Adaptive average pooling** makes the classifier size-independent: the flattened feature
  size is measured with a dummy forward pass in `__init__`, so the same code trains on
  2 classes or 20, at 32×32 or 224×224.
* **`BatchNorm1d` is intentionally avoided** in the head so a batch size of 1 cannot crash
  training.
* The output layer produces raw logits, as `CrossEntropyLoss` expects.

## 15. How the Code Works

### Entry point — `week2_pytorch_image_classification.py`

* `build_parser()` — defines every CLI flag.
* `main()` — parses arguments, dispatches the action and converts expected errors into clear
  messages with exit code `1` (`KeyboardInterrupt` → `130`).
* `command_train()` / `command_evaluate()` / `command_predict()` — one function per action.

### `src/data_loader.py`

* `validate_dataset_dir()` — checks the folder exists, has class sub-folders and that none is
  empty; raises `DatasetError` with a fix-it message.
* `get_transforms()` — returns the augmented *training* pipeline and the deterministic
  *evaluation* pipeline.
* `split_indices()` — shuffles indices with a seeded `torch.Generator`, then slices
  train / validation / test and guarantees each split receives at least one image.
* `create_dataloaders()` — returns a `DataBundle` (three `DataLoader`s + class metadata).
  `split_mode="auto"` uses a `train`/`val`/`test` layout when it is detected and otherwise
  splits the flat dataset randomly; `discover_provided_splits()` and
  `validate_provided_splits()` implement the detection and consistency checks.
* `seed_everything()` — seeds `random`, `numpy` and `torch`.

### `src/model.py`

* `ConvBlock` — two convolutions + BN + ReLU + pooling + dropout.
* `ImageClassifier` — the full CNN; `forward()` returns logits.
* `create_model()` — factory used by training and inference.
* `get_device()` — `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, with
  `mps` support and a safe fallback when CUDA is requested but unavailable.
* `describe_device()`, `count_parameters()`, `check_forward_pass()` — reporting and shape checks.

### `src/train.py`

* `run_one_epoch()` — one pass over a loader; training when an optimiser is passed,
  evaluation otherwise.
* `train_model()` — Adam + `CrossEntropyLoss`, per-epoch printing, deep-copies the weights
  whenever validation accuracy improves, and finally saves the checkpoint containing
  `state_dict`, `class_names`, `num_classes`, `image_size`, `norm_mean`, `norm_std`,
  `model_config`, `training_config` and the full `history`.
* `load_checkpoint()` / `build_model_from_checkpoint()` — restore a model for inference;
  raises a helpful error when the checkpoint is missing or incomplete.

### `src/evaluate.py`

* `collect_predictions()` — inference over the test loader (no gradients).
* `compute_metrics()` — accuracy, macro precision/recall/F1, confusion matrix, classification
  report via `sklearn`.
* `plot_loss_curve()`, `plot_accuracy_curve()`, `plot_confusion_matrix()` — the three figures.
  Matplotlib uses the non-interactive `Agg` backend so the script also works on servers/CI.
* `load_history()` — reads the history from the checkpoint, falling back to
  `outputs/training_history.json`.

### `src/predict.py`

* `load_predictor()` — checkpoint → `(model, class_names, transform, image_size)`, reusing the
  exact normalisation values stored during training.
* `predict_image()` — opens the image, converts it to RGB, applies the transform, runs
  `softmax`, and returns the predicted class, confidence and top-k probabilities.
* `predict_path()` — classifies a single image or every image in a folder.
* `show_image_with_prediction()` — optional preview; skipped automatically on headless systems.

## 16. Troubleshooting

| Problem | Solution |
|---|---|
| `No class sub-folders found inside 'dataset'` | Add one folder per class: `dataset/cats/*.jpg` |
| `No pre-split folders found ... (--split-mode provided)` | You passed `--split-mode provided` but the layout is flat — use `auto` or `random` |
| `The 'val' split has different classes than 'train'` | Every split must contain the same class folders; add the missing ones |
| `These class folders contain no supported images` | The listed folders are empty — add images or delete the folders |
| `Only N image(s) found - at least 3 are needed` | Add more images; 10+ per class is recommended |
| `No saved model found at 'models\image_classifier.pth'` | Run `--train` first |
| `[!] CUDA requested but not available - falling back to CPU` | Expected on a CPU-only machine; use `--device cpu` to silence it |
| `RuntimeError: The size of the tensor ...` | Lower `--batch-size`, or lower `--image-size` |
| Training is slow on Windows | Keep `--num-workers 0`; use `--image-size 64` or a GPU |
| Evaluation complains about missing classes | Use the same `--dataset` you trained with; class folders must match the checkpoint |
| Plot window does not appear | Expected: plots are saved to `outputs/` with the `Agg` backend |
| `ModuleNotFoundError: No module named 'src'` | Run the command from the project root folder |

## 17. Future Improvements

* **Transfer learning** — replace the CNN body with a pretrained ResNet18/EfficientNet and fine-tune.
* **Learning-rate scheduling** — `CosineAnnealingLR` or `OneCycleLR`, plus early stopping.
* **Class weighting / augmentation tuning** — handle imbalanced classes and small datasets.
* **Per-image and per-class metrics** — top-k accuracy, per-class precision/recall breakdown.
* **Normalised confusion matrix** and side-by-side prediction galleries.
* **Experiment tracking** — log every run to CSV/TensorBoard and compare configurations.
* **Config files** — load hyper-parameters from `YAML`/`JSON` instead of flags.
* **A web or desktop UI** — drag-and-drop prediction with a simple interface.
* **Unit tests** — a `tests/` folder with `pytest` covering the split logic and model shapes.
* **ONNX export** — deploy the trained model outside PyTorch.

---

## License

Provided for educational purposes — add your own license before publishing.

## Author

Week 2 – PyTorch Image Classification · AI/ML internship project