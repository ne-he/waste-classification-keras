"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Refactored from the notebook. Works with both the flat baseline CNN and the
nested MobileNetV2 model: when the target Conv2D layer lives inside a nested
backbone, the gradient model is built over the backbone and the remaining head
layers are applied manually (see Pitfall 1 in the execution plan).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.exceptions import ModelError
from src.logger import setup_logger
from src.model import get_last_conv_layer

logger = setup_logger(__name__)

PathLike = Union[str, Path]
_DEFAULT_IMG_SIZE = 224


class GradCAM:
    """Generates Grad-CAM heatmaps for a Keras image classifier.

    Args:
        model: The trained Keras model to explain.
        last_conv_layer_name: Name of the Conv2D layer to use. Auto-detected
            (last Conv2D, searching nested models) when ``None``.

    Raises:
        ModelError: If no suitable Conv2D layer can be located.
    """

    def __init__(self, model: keras.Model, last_conv_layer_name: Optional[str] = None) -> None:
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or get_last_conv_layer(model)
        self._head_layers: list = []
        self._feature_model = self._build_feature_model()
        logger.info("GradCAM initialized on conv layer '%s'", self.last_conv_layer_name)

    def _build_feature_model(self) -> keras.Model:
        """Build a model yielding the conv feature map and a downstream output.

        Returns:
            For a flat model, a model emitting ``[conv_output, predictions]``.
            For a nested backbone, a model emitting ``[conv_output,
            backbone_output]``; the head layers are stored for manual replay.

        Raises:
            ModelError: If the target conv layer cannot be found.
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == self.last_conv_layer_name:
                return keras.Model(self.model.inputs, [layer.output, self.model.output])
            if hasattr(layer, "layers"):
                for sub_layer in layer.layers:
                    if sub_layer.name == self.last_conv_layer_name:
                        self._head_layers = self.model.layers[idx + 1 :]
                        return keras.Model(layer.input, [sub_layer.output, layer.output])
        logger.error("Conv layer '%s' not found", self.last_conv_layer_name)
        raise ModelError(f"Layer not found: {self.last_conv_layer_name}")

    def compute_heatmap(
        self, img_array: np.ndarray, pred_index: Optional[int] = None
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap for a single preprocessed image.

        Args:
            img_array: Image batch of shape ``(1, H, W, 3)``, scaled to [0, 1].
            pred_index: Class index to explain. Defaults to the top prediction.

        Returns:
            A 2-D heatmap with values normalized to [0, 1].
        """
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, downstream = self._feature_model(img_tensor)
            if self._head_layers:
                activations = downstream
                for layer in self._head_layers:
                    activations = layer(activations)
                predictions = activations
            else:
                predictions = downstream

            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_map = conv_outputs[0]
        heatmap = conv_map @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def overlay(self, img_path: PathLike, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Superimpose a heatmap on the original image.

        Args:
            img_path: Path to the original image file.
            heatmap: A 2-D heatmap with values in [0, 1].
            alpha: Blending weight for the heatmap overlay.

        Returns:
            A BGR image array ready to write with ``cv2.imwrite``.

        Raises:
            ModelError: If the image cannot be read.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error("Could not load image: %s", img_path)
            raise ModelError(f"Could not load image: {img_path}")

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    def visualize_batch(self, image_paths: Sequence[PathLike], output_dir: PathLike) -> list[Path]:
        """Generate and save Grad-CAM overlays for multiple images.

        Args:
            image_paths: Paths to the images to explain.
            output_dir: Directory where overlays are written.

        Returns:
            Paths to the saved overlay images.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for img_path in image_paths:
            img_path = Path(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue

            img_resized = cv2.resize(img, (_DEFAULT_IMG_SIZE, _DEFAULT_IMG_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

            heatmap = self.compute_heatmap(img_input)
            overlaid = self.overlay(img_path, heatmap)

            save_path = output_dir / f"gradcam_{img_path.name}"
            cv2.imwrite(str(save_path), overlaid)
            saved.append(save_path)

        logger.info("Saved %d Grad-CAM visualizations to %s", len(saved), output_dir)
        return saved
