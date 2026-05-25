"""Unit tests for :mod:`src.training.callbacks` (skipped without TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for callbacks")

from tensorflow.keras import callbacks as keras_callbacks  # noqa: E402

from src.config import TrainingConfig  # noqa: E402
from src.training.callbacks import get_callbacks  # noqa: E402


def test_get_callbacks_returns_expected_set(tmp_path: Path) -> None:
    """get_callbacks() returns the four standard callbacks."""
    cbs = get_callbacks(TrainingConfig(), tmp_path / "run")
    types = {type(cb) for cb in cbs}
    assert keras_callbacks.EarlyStopping in types
    assert keras_callbacks.ReduceLROnPlateau in types
    assert keras_callbacks.ModelCheckpoint in types
    assert keras_callbacks.CSVLogger in types


def test_get_callbacks_creates_run_directory(tmp_path: Path) -> None:
    """get_callbacks() creates the run directory if it does not exist."""
    run_dir = tmp_path / "new_run"
    get_callbacks(TrainingConfig(), run_dir)
    assert run_dir.is_dir()


def test_early_stopping_uses_configured_patience(tmp_path: Path) -> None:
    """EarlyStopping patience is taken from the config."""
    cbs = get_callbacks(TrainingConfig(early_stopping_patience=9), tmp_path)
    early = next(cb for cb in cbs if isinstance(cb, keras_callbacks.EarlyStopping))
    assert early.patience == 9
