"""Unit tests for :mod:`src.gradcam` (skipped without TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow", reason="TensorFlow required for src.gradcam")

from src.config import ModelConfig  # noqa: E402
from src.exceptions import ModelError  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.model import build_baseline_cnn, build_mobilenet_v2  # noqa: E402

_IMG_SIZE = 32


def test_compute_heatmap_on_flat_cnn() -> None:
    """compute_heatmap() returns a 2-D heatmap normalized to [0, 1]."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"), img_size=_IMG_SIZE)
    gradcam = GradCAM(model)
    img = np.random.rand(1, _IMG_SIZE, _IMG_SIZE, 3).astype(np.float32)
    heatmap = gradcam.compute_heatmap(img)
    assert heatmap.ndim == 2
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0 + 1e-6


def test_gradcam_handles_nested_backbone() -> None:
    """GradCAM locates a conv layer inside a nested MobileNetV2 backbone."""
    model = build_mobilenet_v2(
        ModelConfig(architecture="mobilenet_v2", pretrained=False), img_size=96
    )
    gradcam = GradCAM(model)
    img = np.random.rand(1, 96, 96, 3).astype(np.float32)
    heatmap = gradcam.compute_heatmap(img)
    assert heatmap.ndim == 2


def test_overlay_raises_on_missing_image(tmp_path: Path) -> None:
    """overlay() raises ModelError when the image file cannot be read."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"), img_size=_IMG_SIZE)
    gradcam = GradCAM(model)
    heatmap = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ModelError, match="Could not load image"):
        gradcam.overlay(tmp_path / "missing.jpg", heatmap)


def test_visualize_batch_saves_overlays(tmp_path: Path) -> None:
    """visualize_batch() writes one overlay file per readable input image."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"), img_size=_IMG_SIZE)
    gradcam = GradCAM(model)
    img_paths = []
    for i in range(3):
        path = tmp_path / f"img_{i}.jpg"
        pixels = np.random.randint(0, 256, size=(40, 40, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        img_paths.append(path)
    saved = gradcam.visualize_batch(img_paths, tmp_path / "out")
    assert len(saved) == 3
    assert all(p.exists() for p in saved)
