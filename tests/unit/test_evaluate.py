"""Unit tests for :mod:`src.evaluate` (skipped without TensorFlow)."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for src.evaluate")

from src.config import Config, ModelConfig  # noqa: E402
from src.data import DataLoader  # noqa: E402
from src.evaluate import Evaluator  # noqa: E402
from src.model import build_baseline_cnn, compile_model  # noqa: E402


@pytest.fixture
def evaluator(sample_config: Config) -> Evaluator:
    """Build an Evaluator over a tiny untrained baseline CNN."""
    model = build_baseline_cnn(
        ModelConfig(architecture="baseline_cnn"), img_size=sample_config.data.img_size
    )
    model = compile_model(model, sample_config.training)
    loader = DataLoader(sample_config.data)
    return Evaluator(model, loader, loader.get_class_labels())


def test_evaluate_returns_expected_keys(evaluator: Evaluator) -> None:
    """evaluate() returns accuracy, macro AUC, per-class metrics, and CM."""
    results = evaluator.evaluate()
    assert {"accuracy", "macro_auc", "per_class", "confusion_matrix"} <= results.keys()
    assert 0.0 <= results["accuracy"] <= 1.0


def test_confusion_matrix_shape(evaluator: Evaluator) -> None:
    """The confusion matrix is square with one row/column per class."""
    cm = evaluator.generate_confusion_matrix()
    assert cm.shape == (6, 6)


def test_per_class_auc_includes_macro(evaluator: Evaluator) -> None:
    """compute_per_class_auc() includes a macro-average entry."""
    aucs = evaluator.compute_per_class_auc()
    assert "macro" in aucs
    assert all(0.0 <= v <= 1.0 for v in aucs.values())


def test_full_report_creates_files(evaluator: Evaluator, tmp_path) -> None:
    """generate_full_report() writes a Markdown report and a confusion plot."""
    report_path = evaluator.generate_full_report(tmp_path)
    assert report_path.exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "metrics.json").exists()
