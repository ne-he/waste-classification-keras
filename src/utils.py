"""General-purpose utilities: seeding, filesystem, JSON I/O, timestamps."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np

from src.logger import setup_logger

logger = setup_logger(__name__)

PathLike = Union[str, Path]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries.

    Seeds Python's ``random``, NumPy, the ``PYTHONHASHSEED`` environment
    variable, and TensorFlow (if it is importable).

    Args:
        seed: The integer seed to apply.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not available; skipping tf.random seeding.")

    logger.debug("Random seed set to %d", seed)


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The directory path as a :class:`~pathlib.Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: PathLike) -> None:
    """Serialize a Python object to a JSON file.

    Args:
        data: JSON-serializable object to write.
        path: Destination file path. Parent directories are created.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(path)
    ensure_dir(path.parent)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
    except OSError:
        logger.error("Failed to write JSON to %s", path)
        raise
    logger.debug("Saved JSON to %s", path)


def load_json(path: PathLike) -> Any:
    """Load a Python object from a JSON file.

    Args:
        path: Source file path.

    Returns:
        The deserialized object.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.error("Failed to load JSON from %s", path)
        raise


def timestamp() -> str:
    """Return the current local time as a filename-safe ISO 8601 string.

    Returns:
        A string of the form ``YYYYMMDDTHHMMSS`` (e.g. ``20260518T143012``).
    """
    return datetime.now().strftime("%Y%m%dT%H%M%S")
