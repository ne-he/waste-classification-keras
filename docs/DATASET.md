# Dataset — TrashNet

## Summary

| Property | Value |
|----------|-------|
| Source | https://www.kaggle.com/datasets/feyzazkefe/trashnet |
| Images | 2,527 |
| Classes | cardboard, glass, metal, paper, plastic, trash |
| Original resolution | 384×512 |
| Training resolution | 224×224 |

## Class distribution

| Class | Count | Share |
|-------|-------|-------|
| paper | 594 | 23.5% |
| glass | 501 | 19.8% |
| plastic | 482 | 19.1% |
| metal | 410 | 16.2% |
| cardboard | 403 | 16.0% |
| trash | 137 | 5.4% |

The `trash` class is severely under-represented. The training pipeline
addresses this with balanced class weights (see `docs/TRAINING.md`).

## Layout

The data loader expects six class folders directly under the dataset root:

```
data/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

The dataset is **not committed to git**.

## Acquisition

```bash
python scripts/download_dataset.py
```

The script skips the download if the dataset already exists, attempts a
Kaggle CLI download otherwise, and prints manual instructions as a fallback.
See `data/README.md` for the manual procedure.

## Validation

`DataLoader.validate()` checks that the dataset directory exists, contains
exactly the six expected classes, and that each class has at least 10 images.
It raises `DataError` with a clear message on any failure.
