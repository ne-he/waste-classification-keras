#!/usr/bin/env python
"""CLI entry point for single-image inference.

Usage:
    python scripts/predict.py --image sample.jpg \
        --model models/final/run/model.keras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `import src...` when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.api.inference import InferenceEngine  # noqa: E402
from src.exceptions import ImageClassificationError  # noqa: E402
from src.logger import setup_logger  # noqa: E402

logger = setup_logger("scripts.predict", log_file="predict.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Classify a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", required=True, help="Path to a saved .keras model.")
    return parser.parse_args()


def main() -> int:
    """Run inference on one image and print the result as JSON.

    Returns:
        Process exit code (0 on success, 1 on a handled failure).
    """
    args = parse_args()
    try:
        image_path = Path(args.image)
        if not image_path.exists():
            raise ImageClassificationError(f"Image not found: {image_path}")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ImageClassificationError(f"Could not decode image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        engine = InferenceEngine(args.model)
        result = engine.predict(image_rgb)

        print(json.dumps(result.model_dump(), indent=2))
        return 0
    except ImageClassificationError as exc:
        logger.error("Prediction aborted: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
