# Dataset — TrashNet (dataset-resized)

This project uses the **TrashNet** dataset for waste image classification.

| Property | Value |
|----------|-------|
| Source | https://www.kaggle.com/datasets/feyzazkefe/trashnet |
| Total images | 2,527 |
| Classes (6) | cardboard, glass, metal, paper, plastic, trash |
| Original resolution | 384×512 |

## Expected layout

The data loader expects the following structure under this directory:

```
data/dataset-resized/
├── cardboard/   (403 images)
├── glass/       (501 images)
├── metal/       (410 images)
├── paper/       (594 images)
├── plastic/     (482 images)
└── trash/       (137 images)
```

The dataset itself is **not committed to git** (see `.gitignore`).

## How to download

### Option A — automated (Kaggle API)

```bash
# 1. Install the Kaggle CLI (included in requirements-dev.txt indirectly,
#    or `pip install kaggle`).
# 2. Place your Kaggle API token at ~/.kaggle/kaggle.json
#    (Account → Settings → Create New Token on kaggle.com).
python scripts/download_dataset.py
```

The script checks whether the dataset already exists and skips the download
if so.

### Option B — manual

1. Open https://www.kaggle.com/datasets/feyzazkefe/trashnet
2. Click **Download** to get the zip archive.
3. Extract it so the six class folders land directly under
   `data/dataset-resized/`.
4. Verify with `python scripts/download_dataset.py` (it will validate the
   structure without re-downloading).
