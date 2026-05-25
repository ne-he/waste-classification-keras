# Image Classification — Waste Detection with CNN & Grad-CAM

Production-grade waste image classification on the **TrashNet** dataset
(6 classes, 2,527 images). Refactored from a working Jupyter notebook into a
modular, config-driven, tested Python project with transfer learning
(MobileNetV2), Grad-CAM explainability, a FastAPI inference service, and CLI
tooling.

## Overview

- **Task:** multi-class image classification (cardboard, glass, metal, paper,
  plastic, trash).
- **Models:** a baseline CNN trained from scratch and a MobileNetV2 transfer-
  learning model. The MobileNetV2 transfer-learning model reaches **90.3%**
  validation accuracy over 20 epochs (baseline CNN reference: 54.5%).
- **Explainability:** Grad-CAM heatmaps highlight the image regions driving
  each prediction.
- **Imbalance handling:** balanced class weights target the under-represented
  `trash` class (137 images vs. ~500 for others).

## Quickstart

```bash
# 1. Create an environment (Python 3.12 recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/macOS

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# 3. Get the dataset (see data/README.md)
python scripts/download_dataset.py

# 4. Train, evaluate, predict
python scripts/train.py --config configs/train_mobilenet.yaml
python scripts/evaluate.py --model-path models/final/<run>/model.keras \
    --config configs/train_mobilenet.yaml
python scripts/predict.py --image sample.jpg \
    --model models/final/<run>/model.keras
```

> **Environment note.** The project targets **Python 3.12** with
> **TensorFlow 2.16.1** (the execution plan's pinned `tensorflow==2.14.0`
> does not support Python ≥ 3.12). Training and all TensorFlow-dependent tests
> are run separately in a TensorFlow-equipped environment; unit tests that
> require TensorFlow skip automatically when it is not installed.

## Architecture

```
scripts/ + src/api/        Entry points (CLI + FastAPI)
        │
   src/config.py           YAML configs -> typed dataclasses
        │
   ┌────┴─────┬───────────┬─────────────┐
src/data.py  src/model.py src/evaluate  src/gradcam.py
        │        │         .py          │
   src/training/trainer.py               │
        │                                │
   src/logger.py · src/utils.py · src/exceptions.py
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design rationale.

## Training

Training is config-driven. All hyperparameters live in `configs/*.yaml`:

```bash
python scripts/train.py --config configs/train_mobilenet.yaml
python scripts/train.py --config configs/train_baseline.yaml
```

Artifacts (model, history, config, class index map) are written to a
timestamped directory under `models/final/`. See
[docs/TRAINING.md](docs/TRAINING.md).

## Inference

```bash
python scripts/predict.py --image photo.jpg --model models/final/<run>/model.keras
```

See [docs/INFERENCE.md](docs/INFERENCE.md).

## API

```bash
export MODEL_PATH=models/final/<run>/model.keras
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /health`, `GET /classes`, `POST /predict`, `POST /gradcam`.
See [docs/API.md](docs/API.md).

## Results

MobileNetV2 transfer learning, trained for 20 epochs with balanced class
weights on the TrashNet split (run `mobilenet_v2_20260521T031722`):

| Model | Train Accuracy | Val Accuracy | Val Loss |
|-------|---------------|--------------|----------|
| Baseline CNN (reference) | — | 0.545 | 1.213 |
| MobileNetV2 | 0.915 | **0.903** | 0.281 |

Validation accuracy climbs steadily from 0.73 (epoch 1) to **0.903** (epoch 20)
with no sign of overfitting — the train/val gap stays under 1.5 points. Balanced
class weighting is enabled to lift the under-represented `trash` class.

## Project Structure

```
configs/    YAML configs (base + per-model + inference)
data/       Dataset location (gitignored) + acquisition guide
docs/       Architecture, dataset, training, inference, API docs
models/     Saved models and checkpoints (gitignored)
notebooks/  Original reference notebooks
scripts/    CLI entry points
src/        Library code (data, model, training, evaluation, gradcam, api)
tests/      Unit + integration tests
```

## Testing

```bash
pytest                       # all tests
pytest --cov=src             # with coverage
pytest -m "not slow"         # skip slow tests
```

TensorFlow-dependent tests skip gracefully when TensorFlow is absent. With a
full `requirements-dev.txt` install they all run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching, commit conventions, and
code style.

## License

MIT — see [LICENSE](LICENSE).
