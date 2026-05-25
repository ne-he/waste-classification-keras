"""Integration test for the inference pipeline.

Builds and saves a tiny model, then exercises the InferenceEngine end-to-end.
No training is involved, so this test runs whenever TensorFlow is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for the inference pipeline")

from src.api.inference import DEFAULT_CLASSES, InferenceEngine  # noqa: E402
from src.api.schemas import GradCAMResponse, PredictionResponse  # noqa: E402
from src.config import ModelConfig, TrainingConfig  # noqa: E402
from src.model import build_baseline_cnn, compile_model  # noqa: E402

_IMG_SIZE = 32


@pytest.fixture
def saved_model_path(tmp_path: Path) -> Path:
    """Build, compile, and save a tiny baseline model; return its path."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"), img_size=_IMG_SIZE)
    model = compile_model(model, TrainingConfig())
    path = tmp_path / "model.keras"
    model.save(path)
    return path


@pytest.mark.integration
def test_predict_returns_valid_schema(saved_model_path: Path) -> None:
    """InferenceEngine.predict returns a well-formed PredictionResponse."""
    engine = InferenceEngine(saved_model_path, img_size=_IMG_SIZE)
    image = np.random.randint(0, 256, size=(48, 48, 3), dtype=np.uint8)

    result = engine.predict(image)

    assert isinstance(result, PredictionResponse)
    assert result.predicted_class in DEFAULT_CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.integration
def test_explain_returns_base64_png(saved_model_path: Path) -> None:
    """InferenceEngine.explain returns a GradCAMResponse with a base64 PNG."""
    engine = InferenceEngine(saved_model_path, img_size=_IMG_SIZE)
    image = np.random.randint(0, 256, size=(48, 48, 3), dtype=np.uint8)

    result = engine.explain(image)

    assert isinstance(result, GradCAMResponse)
    assert result.predicted_class in DEFAULT_CLASSES
    assert len(result.heatmap_png_base64) > 0
