"""Unit tests for :mod:`src.config` (pure Python — always runs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.exceptions import ConfigError

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_from_yaml_resolves_inheritance() -> None:
    """A config with `defaults:` inherits and overrides base values."""
    config = Config.from_yaml(_CONFIG_DIR / "train_mobilenet.yaml")
    assert config.model.architecture == "mobilenet_v2"
    assert config.data.img_size == 224  # inherited from base.yaml
    assert config.data.augmentation.rotation_range == 20


def test_baseline_config_loads() -> None:
    """The baseline config loads and selects the baseline architecture."""
    config = Config.from_yaml(_CONFIG_DIR / "train_baseline.yaml")
    assert config.model.architecture == "baseline_cnn"
    assert config.training.epochs == 30


def test_from_yaml_missing_file_raises() -> None:
    """A missing config file raises ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(_CONFIG_DIR / "does_not_exist.yaml")


def test_validate_rejects_unknown_architecture(tmp_path: Path) -> None:
    """An unsupported architecture name raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data:\n  dataset_path: ./d\nmodel:\n  architecture: resnet999\n"
        "training:\n  epochs: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="architecture"):
        Config.from_yaml(bad)


def test_validate_rejects_bad_val_split(tmp_path: Path) -> None:
    """A val_split outside (0, 1) raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data:\n  dataset_path: ./d\n  val_split: 1.5\n"
        "model:\n  architecture: baseline_cnn\ntraining:\n  epochs: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="val_split"):
        Config.from_yaml(bad)


def _write_config(tmp_path: Path, data: str, model: str, training: str) -> Path:
    """Write a minimal config YAML and return its path."""
    path = tmp_path / "cfg.yaml"
    path.write_text(f"data:\n{data}model:\n{model}training:\n{training}", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "data, model, training, match",
    [
        (
            "  dataset_path: ./d\n  img_size: 0\n",
            "  architecture: baseline_cnn\n",
            "  epochs: 1\n",
            "img_size",
        ),
        (
            "  dataset_path: ./d\n  batch_size: 0\n",
            "  architecture: baseline_cnn\n",
            "  epochs: 1\n",
            "batch_size",
        ),
        (
            "  dataset_path: ./d\n",
            "  architecture: baseline_cnn\n  num_classes: 1\n",
            "  epochs: 1\n",
            "num_classes",
        ),
        (
            "  dataset_path: ./d\n",
            "  architecture: baseline_cnn\n  dropout_rate: 1.5\n",
            "  epochs: 1\n",
            "dropout_rate",
        ),
        ("  dataset_path: ./d\n", "  architecture: baseline_cnn\n", "  epochs: 0\n", "epochs"),
        (
            "  dataset_path: ./d\n",
            "  architecture: baseline_cnn\n",
            "  epochs: 1\n  learning_rate: 0\n",
            "learning_rate",
        ),
    ],
)
def test_validate_rejects_bad_values(
    tmp_path: Path, data: str, model: str, training: str, match: str
) -> None:
    """Each invalid field value raises ConfigError naming the field."""
    with pytest.raises(ConfigError, match=match):
        Config.from_yaml(_write_config(tmp_path, data, model, training))


def test_from_yaml_missing_section_raises(tmp_path: Path) -> None:
    """A config missing a required top-level section raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  dataset_path: ./d\ntraining:\n  epochs: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid config"):
        Config.from_yaml(bad)


def test_from_yaml_missing_base_raises(tmp_path: Path) -> None:
    """A `defaults:` pointing at a missing base file raises ConfigError."""
    bad = tmp_path / "child.yaml"
    bad.write_text(
        "defaults: no_such_base.yaml\nmodel:\n  architecture: baseline_cnn\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="Base config not found"):
        Config.from_yaml(bad)


def test_from_yaml_malformed_yaml_raises(tmp_path: Path) -> None:
    """Malformed YAML raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        Config.from_yaml(bad)


def test_to_dict_and_save_round_trip(tmp_path: Path) -> None:
    """to_dict() is nested and save() writes a reloadable YAML file."""
    config = Config.from_yaml(_CONFIG_DIR / "train_mobilenet.yaml")
    as_dict = config.to_dict()
    assert as_dict["model"]["architecture"] == "mobilenet_v2"

    out = tmp_path / "saved.yaml"
    config.save(out)
    reloaded = Config.from_yaml(out)
    assert reloaded.model.architecture == config.model.architecture
    assert reloaded.data.batch_size == config.data.batch_size
