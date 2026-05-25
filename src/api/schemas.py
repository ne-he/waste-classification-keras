"""Pydantic schemas for the inference API.

These models define the request and response contracts for the FastAPI
endpoints. Image bytes are transferred as multipart uploads, so the request
schemas only carry optional tuning parameters.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Optional tuning parameters for a classification request."""

    top_k: int = Field(default=6, ge=1, le=6, description="Number of class scores to return.")


class PredictionResponse(BaseModel):
    """Result of a single image classification."""

    predicted_class: str = Field(description="The top-1 predicted class name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the top class.")
    probabilities: dict[str, float] = Field(description="Per-class probabilities.")


class GradCAMRequest(BaseModel):
    """Optional tuning parameters for a Grad-CAM request."""

    alpha: float = Field(default=0.4, gt=0.0, le=1.0, description="Heatmap blend weight.")


class GradCAMResponse(BaseModel):
    """Result of a Grad-CAM explanation request."""

    predicted_class: str = Field(description="The top-1 predicted class name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the top class.")
    heatmap_png_base64: str = Field(description="Base64-encoded PNG of the overlaid heatmap.")


class HealthResponse(BaseModel):
    """Service health status."""

    status: str = Field(description="Service status, 'ok' when healthy.")
    model_loaded: bool = Field(description="Whether the model is loaded in memory.")
