"""Inference engine wrapping a trained model for API and CLI use.

The :class:`InferenceEngine` loads the model lazily on first use and guards
loading with a lock so it is safe to share across FastAPI worker threads.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from tensorflow import keras

from src.api.schemas import GradCAMResponse, PredictionResponse
from src.exceptions import ModelError
from src.gradcam import GradCAM
from src.logger import setup_logger

logger = setup_logger(__name__)

PathLike = Union[str, Path]

DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
_DEFAULT_IMG_SIZE = 224


class InferenceEngine:
    """Loads a trained model and serves classification / Grad-CAM requests.

    Args:
        model_path: Path to a saved ``.keras`` model file.
        img_size: Square input size the model expects.

    Raises:
        ModelError: If ``model_path`` does not exist.
    """

    def __init__(self, model_path: PathLike, img_size: int = _DEFAULT_IMG_SIZE) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelError(f"Model file not found: {self.model_path}")
        self.img_size = img_size
        self._model: Optional[keras.Model] = None
        self._gradcam: Optional[GradCAM] = None
        self._class_names = self._load_class_names()
        self._lock = threading.Lock()

    def _load_class_names(self) -> list[str]:
        """Load class names from a sibling ``class_indices.json`` if present.

        Returns:
            Class names ordered by index, or the default TrashNet ordering.
        """
        index_file = self.model_path.parent / "class_indices.json"
        if index_file.exists():
            try:
                mapping = json.loads(index_file.read_text(encoding="utf-8"))
                return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Could not read %s: %s; using defaults.", index_file, exc)
        return list(DEFAULT_CLASSES)

    def _ensure_loaded(self) -> None:
        """Load the model and Grad-CAM helper once, in a thread-safe manner.

        Raises:
            ModelError: If the model cannot be loaded.
        """
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                logger.info("Loading model from %s", self.model_path)
                self._model = keras.models.load_model(self.model_path)
            except (OSError, ValueError) as exc:
                logger.error("Failed to load model: %s", exc)
                raise ModelError(f"Failed to load model {self.model_path}: {exc}") from exc
            self._gradcam = GradCAM(self._model)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize an RGB image into a model-ready batch.

        Args:
            image: An RGB image array of shape ``(H, W, 3)``.

        Returns:
            A float32 batch of shape ``(1, img_size, img_size, 3)`` in [0, 1].
        """
        resized = cv2.resize(image, (self.img_size, self.img_size))
        normalized = resized.astype(np.float32)
        if normalized.max() > 1.0:
            normalized /= 255.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, image: np.ndarray) -> PredictionResponse:
        """Classify a single RGB image.

        Args:
            image: An RGB image array of shape ``(H, W, 3)``.

        Returns:
            A :class:`PredictionResponse` with the top class and probabilities.
        """
        self._ensure_loaded()
        assert self._model is not None
        batch = self._preprocess(image)
        probs = self._model.predict(batch, verbose=0)[0]
        top_index = int(np.argmax(probs))
        return PredictionResponse(
            predicted_class=self._class_names[top_index],
            confidence=float(probs[top_index]),
            probabilities={name: float(prob) for name, prob in zip(self._class_names, probs)},
        )

    def explain(self, image: np.ndarray, alpha: float = 0.4) -> GradCAMResponse:
        """Produce a Grad-CAM explanation for a single RGB image.

        Args:
            image: An RGB image array of shape ``(H, W, 3)``.
            alpha: Heatmap blend weight for the overlay.

        Returns:
            A :class:`GradCAMResponse` with the prediction and a base64 PNG.
        """
        self._ensure_loaded()
        assert self._gradcam is not None
        prediction = self.predict(image)

        batch = self._preprocess(image)
        heatmap = self._gradcam.compute_heatmap(batch)

        base = cv2.resize(image, (self.img_size, self.img_size))
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (self.img_size, self.img_size)))
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlaid = cv2.addWeighted(base_bgr, 1 - alpha, heatmap_colored, alpha, 0)

        success, buffer = cv2.imencode(".png", overlaid)
        if not success:
            raise ModelError("Failed to encode Grad-CAM overlay as PNG.")

        return GradCAMResponse(
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            heatmap_png_base64=base64.b64encode(buffer.tobytes()).decode("ascii"),
        )

    @property
    def class_names(self) -> list[str]:
        """The class names served by this engine."""
        return list(self._class_names)

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._model is not None
