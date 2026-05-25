#!/usr/bin/env python
"""CLI script to download (or validate) the TrashNet dataset.

If the dataset already exists locally, the script validates its structure and
exits without downloading. Otherwise it attempts a Kaggle API download, falling
back to printed manual instructions if the Kaggle CLI is unavailable.

Usage:
    python scripts/download_dataset.py [--data-dir data/dataset-resized]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# Allow `import src...` when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__, log_file="download_dataset.log")

KAGGLE_DATASET = "feyzazkefe/trashnet"
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MIN_TOTAL_IMAGES = 2000

MANUAL_INSTRUCTIONS = """
Could not download automatically. Please download the dataset manually:

  1. Open https://www.kaggle.com/datasets/feyzazkefe/trashnet
  2. Click 'Download' to get the zip archive.
  3. Extract it so the six class folders land directly under:
       {data_dir}
  4. Re-run this script to validate.
"""


def is_dataset_present(data_dir: Path) -> bool:
    """Check whether a valid-looking dataset already exists.

    Args:
        data_dir: Directory expected to contain the class subfolders.

    Returns:
        ``True`` if all expected class folders exist and the total image count
        meets the minimum threshold.
    """
    if not data_dir.is_dir():
        return False

    present = {d.name for d in data_dir.iterdir() if d.is_dir()}
    if not set(EXPECTED_CLASSES).issubset(present):
        return False

    total = sum(len(list((data_dir / cls).glob("*.jpg"))) for cls in EXPECTED_CLASSES)
    return total >= MIN_TOTAL_IMAGES


def validate_dataset(data_dir: Path) -> None:
    """Log the per-class image distribution of an existing dataset.

    Args:
        data_dir: Directory containing the class subfolders.
    """
    logger.info("Validating dataset at %s", data_dir)
    total = 0
    for cls in EXPECTED_CLASSES:
        count = len(list((data_dir / cls).glob("*.jpg")))
        total += count
        logger.info("  %-10s: %d images", cls, count)
    logger.info("Total: %d images across %d classes", total, len(EXPECTED_CLASSES))


def download_via_kaggle(data_dir: Path) -> bool:
    """Attempt to download and extract the dataset via the Kaggle CLI.

    Args:
        data_dir: Target directory for the extracted dataset.

    Returns:
        ``True`` on success, ``False`` if the Kaggle CLI is unavailable or the
        download fails.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir.parent / "trashnet.zip"
    try:
        logger.info("Downloading %s via Kaggle CLI...", KAGGLE_DATASET)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir.parent)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Kaggle download failed: %s", exc)
        return False

    archives = list(data_dir.parent.glob("*.zip"))
    if not archives:
        logger.warning("No zip archive found after Kaggle download.")
        return False

    try:
        logger.info("Extracting %s...", archives[0])
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(data_dir.parent)
    except (zipfile.BadZipFile, OSError) as exc:
        logger.error("Extraction failed: %s", exc)
        return False
    finally:
        if zip_path.exists():
            zip_path.unlink()

    return True


def main() -> int:
    """Entry point: parse args, ensure the dataset is present and valid.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(description="Download or validate the TrashNet dataset.")
    parser.add_argument(
        "--data-dir",
        default="data/dataset-resized",
        help="Target dataset directory (default: data/dataset-resized).",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if is_dataset_present(data_dir):
        logger.info("Dataset already present; skipping download.")
        validate_dataset(data_dir)
        return 0

    logger.info("Dataset not found at %s; attempting download.", data_dir)
    if download_via_kaggle(data_dir) and is_dataset_present(data_dir):
        validate_dataset(data_dir)
        logger.info("Dataset downloaded successfully.")
        return 0

    print(MANUAL_INSTRUCTIONS.format(data_dir=data_dir.resolve()))
    return 1


if __name__ == "__main__":
    sys.exit(main())
