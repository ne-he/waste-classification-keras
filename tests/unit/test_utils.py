"""Unit tests for :mod:`src.utils` (pure Python — always runs)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from src.utils import ensure_dir, load_json, save_json, set_seed, timestamp


def test_set_seed_makes_random_deterministic() -> None:
    """set_seed() yields reproducible draws from random and NumPy."""
    set_seed(123)
    first = (random.random(), float(np.random.rand()))
    set_seed(123)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_ensure_dir_creates_nested_directory(tmp_path: Path) -> None:
    """ensure_dir() creates the directory and any missing parents."""
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.is_dir()
    assert result == target


def test_json_round_trip(tmp_path: Path) -> None:
    """save_json() then load_json() returns the original data."""
    data = {"accuracy": 0.77, "classes": ["a", "b"], "n": 3}
    path = tmp_path / "nested" / "metrics.json"
    save_json(data, path)
    assert load_json(path) == data


def test_load_json_missing_file_raises(tmp_path: Path) -> None:
    """load_json() raises OSError when the file does not exist."""
    with pytest.raises(OSError):
        load_json(tmp_path / "missing.json")


def test_load_json_invalid_content_raises(tmp_path: Path) -> None:
    """load_json() raises JSONDecodeError on malformed JSON."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(bad)


def test_save_json_to_unwritable_path_raises(tmp_path: Path) -> None:
    """save_json() raises OSError when the destination cannot be written."""
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        save_json({"a": 1}, not_a_dir / "nested" / "out.json")


def test_timestamp_format() -> None:
    """timestamp() returns a 15-character ISO-like string with a 'T'."""
    stamp = timestamp()
    assert len(stamp) == 15
    assert stamp[8] == "T"
    assert stamp.replace("T", "").isdigit()
