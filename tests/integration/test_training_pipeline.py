"""Integration test for the end-to-end training pipeline.

This test performs a real (tiny) training run. Per the project setup, full
training is executed separately (see README, "Training"), so the test is
marked ``skip`` by default. Remove the skip marker to run it locally.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for the training pipeline")

from src.config import Config  # noqa: E402
from src.data import DataLoader  # noqa: E402
from src.model import build_baseline_cnn, compile_model  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skip(reason="Full training is executed separately; see README 'Training'.")
def test_training_produces_artifacts(sample_config: Config, tmp_path) -> None:
    """A short training run completes and persists all expected artifacts."""
    sample_config.output_dir = str(tmp_path / "runs")
    sample_config.training.epochs = 1

    model = build_baseline_cnn(sample_config.model, img_size=sample_config.data.img_size)
    model = compile_model(model, sample_config.training)
    data_loader = DataLoader(sample_config.data)

    trainer = Trainer(sample_config, model, data_loader)
    history = trainer.train()

    assert "accuracy" in history.history
    assert (trainer.run_dir / "model.keras").exists()
    assert (trainer.run_dir / "history.json").exists()
    assert (trainer.run_dir / "class_indices.json").exists()
