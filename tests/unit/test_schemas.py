"""Unit tests for :mod:`src.api.schemas` (pure Pydantic — always runs)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    GradCAMRequest,
    GradCAMResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def test_prediction_response_valid() -> None:
    """A well-formed PredictionResponse constructs successfully."""
    resp = PredictionResponse(
        predicted_class="metal",
        confidence=0.91,
        probabilities={"metal": 0.91, "glass": 0.09},
    )
    assert resp.predicted_class == "metal"
    assert resp.probabilities["metal"] == pytest.approx(0.91)


def test_prediction_response_rejects_out_of_range_confidence() -> None:
    """Confidence above 1.0 fails validation."""
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_class="metal", confidence=1.5, probabilities={})


def test_prediction_request_defaults_and_bounds() -> None:
    """PredictionRequest defaults top_k to 6 and rejects out-of-range values."""
    assert PredictionRequest().top_k == 6
    with pytest.raises(ValidationError):
        PredictionRequest(top_k=99)


def test_gradcam_request_alpha_bounds() -> None:
    """GradCAMRequest rejects a non-positive alpha."""
    assert GradCAMRequest().alpha == pytest.approx(0.4)
    with pytest.raises(ValidationError):
        GradCAMRequest(alpha=0.0)


def test_gradcam_response_valid() -> None:
    """A well-formed GradCAMResponse constructs successfully."""
    resp = GradCAMResponse(predicted_class="paper", confidence=0.7, heatmap_png_base64="abc123")
    assert resp.heatmap_png_base64 == "abc123"


def test_health_response_valid() -> None:
    """HealthResponse carries a status string and a loaded flag."""
    resp = HealthResponse(status="ok", model_loaded=True)
    assert resp.status == "ok"
    assert resp.model_loaded is True
