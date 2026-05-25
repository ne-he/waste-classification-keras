"""Keras training callbacks assembled from the project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from tensorflow.keras import callbacks as keras_callbacks

from src.config import TrainingConfig
from src.logger import setup_logger

logger = setup_logger(__name__)

PathLike = Union[str, Path]


def get_callbacks(config: TrainingConfig, run_dir: PathLike) -> list[keras_callbacks.Callback]:
    """Build the standard set of training callbacks.

    Assembles EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, and CSVLogger,
    all parameterized from :class:`~src.config.TrainingConfig`.

    Args:
        config: Training configuration (patience, LR schedule parameters).
        run_dir: Directory where the checkpoint and CSV log are written.

    Returns:
        A list of configured Keras callbacks.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[keras_callbacks.Callback] = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras_callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]

    logger.info("Configured %d training callbacks (run_dir=%s)", len(callbacks), run_dir)
    return callbacks
