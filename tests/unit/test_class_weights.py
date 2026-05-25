"""Unit tests for :mod:`src.training.class_weights` (pure NumPy/sklearn)."""

from __future__ import annotations

import numpy as np
import pytest

from src.exceptions import TrainingError
from src.training.class_weights import compute_class_weights


def test_balanced_weights_favor_minority_class() -> None:
    """The rarer class receives a strictly larger weight."""
    labels = np.array([0, 0, 0, 0, 0, 0, 1])
    weights = compute_class_weights(labels)
    assert weights[1] > weights[0]


def test_weights_keyed_by_int_class_index() -> None:
    """Returned weights are keyed by plain Python ints with float values."""
    weights = compute_class_weights(np.array([0, 1, 2, 0, 1, 2]))
    assert set(weights.keys()) == {0, 1, 2}
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in weights.items())


def test_balanced_dataset_yields_uniform_weights() -> None:
    """A perfectly balanced dataset yields equal weights of 1.0."""
    weights = compute_class_weights(np.array([0, 0, 1, 1, 2, 2]))
    assert all(w == pytest.approx(1.0) for w in weights.values())


def test_empty_labels_raise_training_error() -> None:
    """An empty label array raises TrainingError."""
    with pytest.raises(TrainingError, match="empty"):
        compute_class_weights(np.array([]))
