#!/usr/bin/env python
"""CLI entry point for generating Grad-CAM visualizations.

Usage:
    python scripts/generate_gradcam.py --model-path models/final/run/model.keras \
        --image-dir data/dataset-resized/metal --output outputs/gradcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `import src...` when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow import keras  # noqa: E402

from src.exceptions import ImageClassificationError, ModelError  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.logger import setup_logger  # noqa: E402

logger = setup_logger("scripts.generate_gradcam", log_file="gradcam.log")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap overlays.")
    parser.add_argument("--model-path", required=True, help="Path to a saved .keras model.")
    parser.add_argument("--image-dir", required=True, help="Directory of input images.")
    parser.add_argument(
        "--output",
        default="outputs/gradcam",
        help="Output directory for overlays (default: outputs/gradcam).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Maximum number of images to process (default: 12).",
    )
    return parser.parse_args()


def main() -> int:
    """Load a model and generate Grad-CAM overlays for a directory of images.

    Returns:
        Process exit code (0 on success, 1 on a handled failure).
    """
    args = parse_args()
    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")
        try:
            model = keras.models.load_model(model_path)
        except (OSError, ValueError) as exc:
            raise ModelError(f"Failed to load model {model_path}: {exc}") from exc

        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise ModelError(f"Image directory not found: {image_dir}")
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )[: args.limit]
        if not image_paths:
            raise ModelError(f"No images found in {image_dir}")

        gradcam = GradCAM(model)
        saved = gradcam.visualize_batch(image_paths, args.output)

        print(f"Saved {len(saved)} Grad-CAM overlays to: {args.output}")
        return 0
    except ImageClassificationError as exc:
        logger.error("Grad-CAM generation aborted: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
