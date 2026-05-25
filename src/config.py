"""Configuration management using dataclasses and YAML.

Configs are layered: an env-specific YAML file may declare ``defaults:`` to
inherit from a base file. The merged result is validated and materialized into
typed dataclasses, so the rest of the codebase never touches raw dictionaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

from src.exceptions import ConfigError
from src.logger import setup_logger

logger = setup_logger(__name__)

PathLike = Union[str, Path]

SUPPORTED_ARCHITECTURES = ("baseline_cnn", "mobilenet_v2")


@dataclass
class AugmentationConfig:
    """Image augmentation parameters applied to the training set only."""

    rotation_range: int = 20
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    width_shift_range: float = 0.0
    height_shift_range: float = 0.0


@dataclass
class DataConfig:
    """Dataset loading and splitting parameters."""

    dataset_path: str
    img_size: int = 224
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 42
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    architecture: str
    num_classes: int = 6
    dropout_rate: float = 0.5
    pretrained: bool = True
    freeze_backbone: bool = True


@dataclass
class TrainingConfig:
    """Training loop and callback parameters."""

    epochs: int = 20
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    use_class_weights: bool = True
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class Config:
    """Top-level configuration aggregating data, model, and training sections."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output_dir: str = "./models/final"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: PathLike) -> "Config":
        """Load and validate a :class:`Config` from a YAML file.

        Supports single-level inheritance via a top-level ``defaults:`` key
        pointing to a sibling YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A fully validated :class:`Config` instance.

        Raises:
            ConfigError: If the file is missing, malformed, or invalid.

        Example:
            >>> cfg = Config.from_yaml("configs/train_mobilenet.yaml")
        """
        path = Path(path)
        if not path.exists():
            logger.error("Config file not found: %s", path)
            raise ConfigError(f"Config file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse YAML config %s: %s", path, exc)
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if "defaults" in raw:
            base_name = raw.pop("defaults")
            base_path = path.parent / base_name
            if not base_path.exists():
                raise ConfigError(f"Base config not found: {base_path}")
            try:
                with open(base_path, "r", encoding="utf-8") as handle:
                    base_raw = yaml.safe_load(handle) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"Invalid YAML in {base_path}: {exc}") from exc
            raw = cls._deep_merge(base_raw, raw)

        try:
            data_raw = dict(raw["data"])
            aug_raw = data_raw.pop("augmentation", {})
            config = cls(
                data=DataConfig(augmentation=AugmentationConfig(**aug_raw), **data_raw),
                model=ModelConfig(**raw["model"]),
                training=TrainingConfig(**raw["training"]),
                output_dir=raw.get("output_dir", "./models/final"),
                log_level=raw.get("log_level", "INFO"),
            )
        except (KeyError, TypeError) as exc:
            logger.error("Invalid config structure in %s: %s", path, exc)
            raise ConfigError(f"Invalid config: {exc}") from exc

        config.validate()
        logger.info("Loaded config from %s (architecture=%s)", path, config.model.architecture)
        return config

    def validate(self) -> None:
        """Validate field values and cross-field constraints.

        Raises:
            ConfigError: If any field holds an invalid value.
        """
        if self.model.architecture not in SUPPORTED_ARCHITECTURES:
            raise ConfigError(
                f"Unsupported architecture '{self.model.architecture}'. "
                f"Expected one of {SUPPORTED_ARCHITECTURES}."
            )
        if not 0.0 < self.data.val_split < 1.0:
            raise ConfigError(f"val_split must be in (0, 1), got {self.data.val_split}")
        if self.data.img_size <= 0:
            raise ConfigError(f"img_size must be positive, got {self.data.img_size}")
        if self.data.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.data.batch_size}")
        if self.model.num_classes <= 1:
            raise ConfigError(f"num_classes must be > 1, got {self.model.num_classes}")
        if not 0.0 <= self.model.dropout_rate < 1.0:
            raise ConfigError(f"dropout_rate must be in [0, 1), got {self.model.dropout_rate}")
        if self.training.epochs <= 0:
            raise ConfigError(f"epochs must be positive, got {self.training.epochs}")
        if self.training.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.training.learning_rate}")

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge ``override`` into ``base`` without mutating inputs.

        Args:
            base: The base dictionary (lower precedence).
            override: The overriding dictionary (higher precedence).

        Returns:
            A new merged dictionary.
        """
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a nested plain dictionary.

        Returns:
            A dictionary representation suitable for serialization.
        """
        return asdict(self)

    def save(self, path: PathLike) -> None:
        """Write the config to a YAML file.

        Args:
            path: Destination file path.

        Raises:
            ConfigError: If the file cannot be written.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(self.to_dict(), handle, default_flow_style=False, sort_keys=False)
        except OSError as exc:
            logger.error("Failed to save config to %s: %s", path, exc)
            raise ConfigError(f"Could not save config to {path}: {exc}") from exc
        logger.debug("Saved config to %s", path)
