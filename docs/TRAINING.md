# Training

## Run

```bash
python scripts/train.py --config configs/train_mobilenet.yaml
python scripts/train.py --config configs/train_baseline.yaml --output-dir models/exp1
```

`--config` is required; `--output-dir` optionally overrides the config's
output directory.

## Configuration

All hyperparameters live in `configs/*.yaml`. `train_mobilenet.yaml` and
`train_baseline.yaml` inherit shared defaults from `base.yaml`:

| Section | Key settings |
|---------|--------------|
| `data` | `img_size`, `batch_size`, `val_split`, `seed`, `augmentation` |
| `model` | `architecture`, `dropout_rate`, `pretrained`, `freeze_backbone` |
| `training` | `epochs`, `learning_rate`, `optimizer`, `use_class_weights`, callback patience |

## Pipeline

`Trainer.train()` performs:

1. Dataset validation (`DataLoader.validate`).
2. Generator construction (deterministic 80/20 split).
3. Balanced class-weight computation when `use_class_weights` is true.
4. `model.fit` with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, and
   CSVLogger callbacks.
5. Artifact persistence into `models/final/<arch>_<timestamp>/`:
   `model.keras`, `history.json`, `config.yaml`, `class_indices.json`.

## Class weighting

`trash` has only 137 images. `compute_class_weights` applies scikit-learn's
`'balanced'` heuristic so rare-class errors are penalized more heavily — the
key lever for lifting the `trash` F1-score from ~0.39 toward ≥ 0.50.

## Reproducibility

`scripts/train.py` calls `set_seed(config.data.seed)` before building any
component. Data splits use the same seed, so runs are reproducible.

## Targets

| Model | Val accuracy target |
|-------|--------------------|
| Baseline CNN | ≥ 0.50 (notebook: 0.545) |
| MobileNetV2 | ≥ 0.75 (notebook: 0.771) |

## Environment note

Training requires TensorFlow (2.16.1) and is run in a TensorFlow-equipped
Python 3.12 environment. The end-to-end training integration test
(`tests/integration/test_training_pipeline.py`) is `skip`-marked by default;
remove the marker to exercise a tiny training run locally.
