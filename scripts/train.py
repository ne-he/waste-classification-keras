#!/usr/bin/env python
"""CLI entry point for training a waste classification model.

Usage:
    python scripts/train.py --config configs/train_mobilenet.yaml
    python scripts/train.py --config configs/train_baseline.yaml --output-dir models/exp1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `import src...` when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402
from src.data import DataLoader  # noqa: E402
from src.exceptions import ImageClassificationError  # noqa: E402
from src.logger import setup_logger  # noqa: E402
from src.model import build_model, compile_model  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils import set_seed  # noqa: E402

logger = setup_logger("scripts.train", log_file="training.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train a waste classification model.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory from the config.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the model and run training end-to-end.

    Returns:
        Process exit code (0 on success, 1 on a handled failure).
    """
    args = parse_args()
    try:
        import tensorflow as tf  # noqa: PLC0415

        config = Config.from_yaml(args.config)
        if args.output_dir:
            config.output_dir = args.output_dir

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s) — fast training mode")
        else:
            print("⚠️ No GPU detected — CPU mode: reducing batch_size to 16 (~6 min/epoch)")
            config.data.batch_size = 16

        set_seed(config.data.seed)

        data_loader = DataLoader(config.data)
        model = build_model(config.model, img_size=config.data.img_size)
        model = compile_model(model, config.training)

        trainer = Trainer(config, model, data_loader)
        history = trainer.train()

        best_val_acc = max(history.history.get("val_accuracy", [0.0]))
        print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
        print(f"Artifacts saved to: {trainer.run_dir}")
        return 0
    except ImageClassificationError as exc:
        logger.error("Training aborted: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
