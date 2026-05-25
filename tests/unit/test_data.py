"""Unit tests for :mod:`src.data`.

The whole module is skipped when TensorFlow is unavailable, since
:mod:`src.data` imports Keras at module load time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for src.data")

from src.config import Config  # noqa: E402
from src.data import DataLoader  # noqa: E402
from src.exceptions import DataError  # noqa: E402


def test_validate_accepts_valid_dataset(sample_config: Config) -> None:
    """validate() passes silently on a well-formed dataset."""
    loader = DataLoader(sample_config.data)
    loader.validate()  # should not raise


def test_validate_raises_on_missing_directory(sample_config: Config) -> None:
    """validate() raises DataError when the dataset path is absent."""
    sample_config.data.dataset_path = "/nonexistent/path/xyz"
    loader = DataLoader(sample_config.data)
    with pytest.raises(DataError, match="does not exist"):
        loader.validate()


def test_validate_raises_on_too_few_images(sample_config: Config, tmp_path: Path) -> None:
    """validate() raises DataError when a class has too few images."""
    for cls in DataLoader.EXPECTED_CLASSES:
        (tmp_path / cls).mkdir()
    sample_config.data.dataset_path = str(tmp_path)
    loader = DataLoader(sample_config.data)
    with pytest.raises(DataError, match="images"):
        loader.validate()


def test_class_distribution_counts_all_classes(sample_config: Config) -> None:
    """get_class_distribution() returns a count for every class."""
    loader = DataLoader(sample_config.data)
    dist = loader.get_class_distribution()
    assert set(dist.keys()) == set(DataLoader.EXPECTED_CLASSES)
    assert all(count == 12 for count in dist.values())


def test_generators_respect_augmentation_params(sample_config: Config) -> None:
    """get_generators() builds train/val iterators with the configured split."""
    loader = DataLoader(sample_config.data)
    train_gen, val_gen = loader.get_generators()
    total = train_gen.samples + val_gen.samples
    assert total == 6 * 12
    assert val_gen.samples == pytest.approx(total * 0.25, abs=6)
    assert len(loader.get_class_labels()) == 6
