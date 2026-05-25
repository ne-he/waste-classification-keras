#!/usr/bin/env python
"""CLI entry point for evaluating a trained model.

Usage:
    python scripts/evaluate.py --model-path models/final/run/model.keras \
        --config configs/train_mobilenet.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `import src...` when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow import keras  # noqa: E402

from src.config import Config  # noqa: E402
from src.data import DataLoader  # noqa: E402
from src.evaluate import Evaluator  # noqa: E402
from src.exceptions import ImageClassificationError, ModelError  # noqa: E402
from src.logger import setup_logger  # noqa: E402
from src.utils import set_seed  # noqa: E402

logger = setup_logger("scripts.evaluate", log_file="evaluate.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model-path", required=True, help="Path to a saved .keras model.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory for the report and plots (default: outputs/evaluation).",
    )
    return parser.parse_args()


def main() -> int:
    """Load a model, evaluate it, and write a full report.

    Returns:
        Process exit code (0 on success, 1 on a handled failure).
    """
    args = parse_args()
    try:
        config = Config.from_yaml(args.config)
        set_seed(config.data.seed)

        model_path = Path(args.model_path)
        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")
        try:
            model = keras.models.load_model(model_path)
        except (OSError, ValueError) as exc:
            raise ModelError(f"Failed to load model {model_path}: {exc}") from exc

        data_loader = DataLoader(config.data)
        data_loader.validate()
        class_names = data_loader.get_class_labels()

        evaluator = Evaluator(model, data_loader, class_names)
        report_path = evaluator.generate_full_report(args.output_dir)

        print(f"Evaluation report written to: {report_path}")
        return 0
    except ImageClassificationError as exc:
        logger.error("Evaluation aborted: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
