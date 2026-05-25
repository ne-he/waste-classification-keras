"""End-to-end training pipeline.

The :class:`Trainer` orchestrates dataset validation, class-weight computation,
callback setup, model fitting, and artifact persistence into a timestamped run
directory under ``config.output_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tensorflow import keras

from src.config import Config
from src.data import DataLoader
from src.exceptions import TrainingError
from src.logger import setup_logger
from src.training.callbacks import get_callbacks
from src.training.class_weights import compute_class_weights
from src.utils import save_json, timestamp

logger = setup_logger(__name__, log_file="training.log")


class Trainer:
    """Coordinates the full training run for a single model.

    Args:
        config: The complete project configuration.
        model: A **compiled** Keras model ready for ``fit``.
        data_loader: Data loader providing train/validation generators.
    """

    def __init__(self, config: Config, model: keras.Model, data_loader: DataLoader) -> None:
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.run_dir: Path = Path(config.output_dir) / f"{config.model.architecture}_{timestamp()}"

    def train(self) -> keras.callbacks.History:
        """Run the training loop and persist artifacts.

        Validates the dataset, builds generators, optionally computes balanced
        class weights, fits the model with callbacks, and saves all artifacts.

        Returns:
            The Keras :class:`~keras.callbacks.History` from ``model.fit``.

        Raises:
            TrainingError: If dataset validation or model fitting fails.
        """
        self.data_loader.validate()
        train_gen, val_gen = self.data_loader.get_generators()

        class_weight: Optional[dict[int, float]] = None
        if self.config.training.use_class_weights:
            class_weight = compute_class_weights(self.data_loader.get_y_train())
        else:
            logger.info("Class weighting disabled by config.")

        callbacks = get_callbacks(self.config.training, self.run_dir)

        logger.info(
            "Starting training: %s for %d epochs",
            self.config.model.architecture,
            self.config.training.epochs,
        )
        try:
            history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=self.config.training.epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("Training failed: %s", exc)
            raise TrainingError(f"Model training failed: {exc}") from exc

        logger.info("Training complete.")
        self.save_artifacts(history)
        return history

    def save_artifacts(self, history: keras.callbacks.History) -> Path:
        """Persist the model, training history, config, and class mapping.

        Args:
            history: The training history returned by :meth:`train`.

        Returns:
            The run directory containing all saved artifacts.

        Raises:
            TrainingError: If any artifact cannot be written.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save(self.run_dir / "model.keras")
            save_json(history.history, self.run_dir / "history.json")
            self.config.save(self.run_dir / "config.yaml")
            class_indices = self.data_loader.get_generators()[0].class_indices
            save_json(class_indices, self.run_dir / "class_indices.json")
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to save training artifacts: %s", exc)
            raise TrainingError(f"Could not save artifacts: {exc}") from exc

        logger.info("Saved training artifacts to %s", self.run_dir)
        return self.run_dir
