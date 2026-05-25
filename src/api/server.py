"""FastAPI inference service for waste image classification.

Endpoints:
    GET  /health   — service and model status
    GET  /classes  — list of class names
    POST /predict  — classify an uploaded image
    POST /gradcam  — Grad-CAM explanation for an uploaded image

The model path is read from the ``MODEL_PATH`` environment variable. The model
itself is loaded lazily on the first prediction request.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceEngine
from src.api.schemas import GradCAMResponse, HealthResponse, PredictionResponse
from src.exceptions import ImageClassificationError
from src.logger import setup_logger

logger = setup_logger(__name__, log_file="api.log")

_DEFAULT_MODEL_PATH = "models/final/mobilenet_v2_latest/model.keras"
_INFERENCE_CONFIG = Path("configs/inference.yaml")

_engine: Optional[InferenceEngine] = None


def _cors_origins() -> list[str]:
    """Read allowed CORS origins from the inference config.

    Returns:
        A list of allowed origins, defaulting to ``["*"]``.
    """
    if _INFERENCE_CONFIG.exists():
        try:
            raw = yaml.safe_load(_INFERENCE_CONFIG.read_text(encoding="utf-8")) or {}
            return list(raw.get("api", {}).get("cors_origins", ["*"]))
        except yaml.YAMLError as exc:
            logger.warning("Could not parse %s: %s", _INFERENCE_CONFIG, exc)
    return ["*"]


def get_engine() -> InferenceEngine:
    """Return the shared inference engine, creating it on first use.

    Returns:
        The process-wide :class:`InferenceEngine`.

    Raises:
        HTTPException: 503 if the model file is unavailable.
    """
    global _engine
    if _engine is None:
        model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
        try:
            _engine = InferenceEngine(model_path)
        except ImageClassificationError as exc:
            logger.error("Engine initialization failed: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _engine


def _read_image(upload: UploadFile, raw: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB array.

    Args:
        upload: The uploaded file (used only for its filename in errors).
        raw: The raw image bytes.

    Returns:
        An RGB image array of shape ``(H, W, 3)``.

    Raises:
        HTTPException: 400 if the bytes cannot be decoded as an image.
    """
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


app = FastAPI(title="Waste Classification API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service health and whether the model is loaded."""
    loaded = _engine is not None and _engine.is_loaded
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get("/classes")
def classes() -> dict[str, list[str]]:
    """Return the list of class names served by the model."""
    return {"classes": get_engine().class_names}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded image and return per-class probabilities."""
    raw = await file.read()
    image = _read_image(file, raw)
    try:
        return get_engine().predict(image)
    except ImageClassificationError as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/gradcam", response_model=GradCAMResponse)
async def gradcam(file: UploadFile = File(...)) -> GradCAMResponse:
    """Return a Grad-CAM heatmap overlay for an uploaded image."""
    raw = await file.read()
    image = _read_image(file, raw)
    try:
        return get_engine().explain(image)
    except ImageClassificationError as exc:
        logger.error("Grad-CAM failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
