"""Standardized logging setup for the project.

Every module in :mod:`src` obtains its logger via :func:`setup_logger`, which
attaches a console handler and (optionally) a rotating-free file handler with a
consistent format. ``print`` is never used inside ``src``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_LOG_DIR = Path("logs")


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
) -> logging.Logger:
    """Create or retrieve a configured logger.

    The logger is configured only once per name; repeated calls return the same
    logger without duplicating handlers.

    Args:
        name: Logger name, typically the calling module's ``__name__``.
        log_file: Optional file name (relative to the ``logs/`` directory) to
            also write logs to. If ``None``, only console logging is used.
        level: Logging level as a string (e.g. ``"INFO"``, ``"DEBUG"``).

    Returns:
        A configured :class:`logging.Logger` instance.

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("data loaded")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level.upper())
    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_DEFAULT_LOG_DIR / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
