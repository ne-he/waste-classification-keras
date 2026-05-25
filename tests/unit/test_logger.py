"""Unit tests for :mod:`src.logger` (pure Python — always runs)."""

from __future__ import annotations

import logging

from src.logger import setup_logger


def test_setup_logger_returns_logger() -> None:
    """setup_logger() returns a Logger with at least a console handler."""
    logger = setup_logger("test.logger.console")
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) >= 1


def test_setup_logger_is_idempotent() -> None:
    """Calling setup_logger() twice does not duplicate handlers."""
    first = setup_logger("test.logger.idempotent")
    handler_count = len(first.handlers)
    second = setup_logger("test.logger.idempotent")
    assert first is second
    assert len(second.handlers) == handler_count


def test_setup_logger_respects_level() -> None:
    """setup_logger() applies the requested logging level."""
    logger = setup_logger("test.logger.level", level="DEBUG")
    assert logger.level == logging.DEBUG


def test_setup_logger_writes_file(tmp_path, monkeypatch) -> None:
    """When log_file is given, a file handler writing to logs/ is attached."""
    monkeypatch.chdir(tmp_path)
    logger = setup_logger("test.logger.file", log_file="unit.log")
    logger.info("hello")
    assert (tmp_path / "logs" / "unit.log").exists()
