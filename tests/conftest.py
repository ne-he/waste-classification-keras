"""Shared pytest fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure `import src...` works when pytest is invoked from any directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    AugmentationConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGES_PER_CLASS = 12


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny on-disk dataset (12 random images per class).

    Args:
        tmp_path_factory: Pytest factory for session-scoped temp directories.

    Returns:
        Path to the dataset root containing the six class folders.
    """
    root = tmp_path_factory.mktemp("dataset-resized")
    rng = np.random.default_rng(42)
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(IMAGES_PER_CLASS):
            pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(cls_dir / f"{cls}_{i}.jpg")
    return root


@pytest.fixture
def sample_config(tiny_dataset: Path) -> Config:
    """Return a minimal valid :class:`Config` pointing at the tiny dataset.

    Args:
        tiny_dataset: Path fixture providing a small on-disk dataset.

    Returns:
        A fully populated :class:`Config` instance.
    """
    return Config(
        data=DataConfig(
            dataset_path=str(tiny_dataset),
            img_size=32,
            batch_size=4,
            val_split=0.25,
            seed=42,
            augmentation=AugmentationConfig(),
        ),
        model=ModelConfig(architecture="mobilenet_v2", num_classes=6),
        training=TrainingConfig(epochs=1),
    )
