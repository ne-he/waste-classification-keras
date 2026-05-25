"""Custom exception hierarchy for the image classification project.

All project-specific errors inherit from :class:`ImageClassificationError`,
which makes it possible to catch any project error with a single ``except``.
"""

from __future__ import annotations


class ImageClassificationError(Exception):
    """Base exception for all errors raised by this project."""


class ConfigError(ImageClassificationError):
    """Raised when configuration loading or validation fails."""


class DataError(ImageClassificationError):
    """Raised when dataset loading or validation fails."""


class ModelError(ImageClassificationError):
    """Raised when model construction, loading, or inspection fails."""


class TrainingError(ImageClassificationError):
    """Raised when the training pipeline encounters an unrecoverable error."""
