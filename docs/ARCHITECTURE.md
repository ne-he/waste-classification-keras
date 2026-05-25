# Architecture

## Goals

Refactor a monolithic, Colab-bound notebook into a modular, testable, and
config-driven Python project without changing model behavior.

## Layers

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Entry | `scripts/*.py`, `src/api/` | CLI parsing, orchestration, HTTP |
| Config | `src/config.py` | Load/validate YAML into typed dataclasses |
| Data | `src/data.py` | Loading, splitting, augmentation, validation |
| Model | `src/model.py` | Architectures + factory + Grad-CAM layer lookup |
| Training | `src/training/` | Class weights, callbacks, training loop |
| Evaluation | `src/evaluate.py` | Metrics, confusion matrix, AUC, reports |
| Explainability | `src/gradcam.py` | Grad-CAM heatmaps |
| Cross-cutting | `src/logger.py`, `src/utils.py`, `src/exceptions.py` | Logging, helpers, errors |

## Design decisions

- **Dataclasses over Pydantic for config.** Keeps runtime dependencies lean;
  Pydantic is used only for the API request/response contracts.
- **Config inheritance.** Env-specific YAML files declare `defaults:` to merge
  over `base.yaml`, avoiding duplication.
- **Lazy, thread-safe model loading.** `InferenceEngine` loads the model on
  first request behind a lock, so it is safe across FastAPI worker threads.
- **Custom exception hierarchy.** All errors derive from
  `ImageClassificationError`, so callers can catch project errors precisely.
- **Reproducibility.** `set_seed` seeds Python, NumPy, and TensorFlow; data
  splits are deterministic (`seed=42`).

## Known migration path

The data pipeline uses Keras' `ImageDataGenerator`, which is deprecated in
favor of `tf.keras.utils.image_dataset_from_directory`. It is retained for
parity with the original notebook. A future change can swap the loader behind
the stable `DataLoader` interface without touching callers.

## Environment

The execution plan pinned `tensorflow==2.14.0` / Python 3.10. Because the
available runtime is Python 3.12, the project uses **TensorFlow 2.16.1**, the
nearest version supporting Python 3.12. The public APIs are unaffected.
