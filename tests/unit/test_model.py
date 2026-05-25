"""Unit tests for :mod:`src.model`.

Skipped entirely when TensorFlow is unavailable. MobileNetV2 is built with
``pretrained=False`` to avoid network access during testing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for src.model")

from src.config import ModelConfig, TrainingConfig  # noqa: E402
from src.exceptions import ModelError  # noqa: E402
from src.model import (  # noqa: E402
    build_baseline_cnn,
    build_mobilenet_v2,
    build_model,
    compile_model,
    get_last_conv_layer,
)


def test_baseline_cnn_builds_with_correct_output_shape() -> None:
    """The baseline CNN outputs a 6-class probability vector."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"))
    assert model.output_shape == (None, 6)
    # Notebook reference: ~11.17M parameters.
    assert 10_000_000 < model.count_params() < 13_000_000


def test_mobilenet_v2_builds_with_correct_output_shape() -> None:
    """The MobileNetV2 model outputs a 6-class probability vector."""
    model = build_mobilenet_v2(ModelConfig(architecture="mobilenet_v2", pretrained=False))
    assert model.output_shape == (None, 6)
    # Notebook reference: ~2.42M parameters.
    assert 2_000_000 < model.count_params() < 3_000_000


def test_build_model_dispatches_by_architecture() -> None:
    """build_model() selects the builder named by config.architecture."""
    cnn = build_model(ModelConfig(architecture="baseline_cnn"))
    mnet = build_model(ModelConfig(architecture="mobilenet_v2", pretrained=False))
    assert cnn.name == "Baseline_CNN"
    assert mnet.name == "MobileNetV2_Transfer"


def test_build_model_rejects_unknown_architecture() -> None:
    """build_model() raises ModelError for an unsupported architecture."""
    bad = ModelConfig(architecture="baseline_cnn")
    bad.architecture = "resnet999"
    with pytest.raises(ModelError, match="Unknown architecture"):
        build_model(bad)


def test_compile_model_sets_optimizer_and_loss() -> None:
    """compile_model() attaches an optimizer and a loss function."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"))
    compiled = compile_model(model, TrainingConfig(learning_rate=1e-3))
    assert compiled.optimizer is not None
    assert compiled.loss == "categorical_crossentropy"


def test_compile_model_rejects_unknown_optimizer() -> None:
    """compile_model() raises ModelError for an unsupported optimizer."""
    model = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"))
    with pytest.raises(ModelError, match="Unknown optimizer"):
        compile_model(model, TrainingConfig(optimizer="adagrad999"))


def test_get_last_conv_layer_finds_conv() -> None:
    """get_last_conv_layer() locates a Conv2D layer in both architectures."""
    cnn = build_baseline_cnn(ModelConfig(architecture="baseline_cnn"))
    mnet = build_mobilenet_v2(ModelConfig(architecture="mobilenet_v2", pretrained=False))
    assert isinstance(get_last_conv_layer(cnn), str)
    assert isinstance(get_last_conv_layer(mnet), str)
