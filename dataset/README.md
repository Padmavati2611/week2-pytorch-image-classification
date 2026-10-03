# Dataset folder

This folder contains the **Dogs vs Cats** subset used by this project:

```text
dataset/
├── train/
│   ├── cats/     500 images
│   └── dogs/     500 images
├── val/
│   ├── cats/     100 images
│   └── dogs/     100 images
├── README.md
└── dogs_vs_cats_subset/
    └── dogs_vs_cats_subset/
        ├── CDLA-Permissive-2.0.pdf
        └── test/     100 images WITHOUT labels (1.jpg ... 100.jpg)
```

Class names (`cats`, `dogs`) are detected automatically from the folder names — nothing is
hard-coded. The images come from the Kaggle Dogs vs Cats dataset
(licence: CDLA-Permissive-2.0, see the PDF above).

---

## Two supported layouts

### 1. Pre-split dataset (what this project uses)

The `train/` and `val/` folders are supplied by you and are used **as-is** — the project
does not re-shuffle them:

```text
dataset/
├── train/
│   ├── <class_a>/*.jpg
│   └── <class_b>/*.jpg
├── val/
│   ├── <class_a>/*.jpg
│   └── <class_b>/*.jpg
└── test/            (optional)
    ├── <class_a>/*.jpg
    └── <class_b>/*.jpg
```

Because there is no labelled `test/` folder here, **50 % of `val/` (100 images) is held out
automatically as the test split** so the model can still be evaluated honestly. Control this
with `--val-test-ratio` (default `0.5`).

### 2. Flat dataset (the default `ImageFolder` layout)

One folder per class, split automatically 70 / 15 / 15 by the project:

```text
dataset/
├── class_1/*.jpg
├── class_2/*.jpg
└── ...
```

Use this if you want the project to create the split for you. A directory is treated as
pre-split only when it contains `train` plus `val`/`test` folders that themselves contain
class folders — so a class that happens to be named `train` is still handled correctly.

---

## Rules

* **Folder name = class name**, kept exactly as written (`cats`, `dogs`, `class_1`, ...).
* **Supported extensions:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`.
* Stray non-image files inside a class folder (`.txt`, `.DS_Store`, ...) are ignored.
* Empty class folders raise an error instead of being silently ignored.
* Every split must contain **the same class folders**, otherwise you get a clear error.
* Images may have different sizes; they are resized to `--image-size` and normalised.
* Minimum 3 images in total. **10+ images per class** is a good minimum for a meaningful
  validation/test score.

## Using the unlabelled `test/` images

Those 100 images have no class information in the filename, so they cannot be used to score
the model. They are useful for demonstrating batch prediction:

```bash
python week2_pytorch_image_classification.py --predict dataset/dogs_vs_cats_subset/dogs_vs_cats_subset/test
```

## Adding your own dataset

Replace the contents of `dataset/` with your own class folders. No code change is needed —
class names and the number of classes are detected at runtime.