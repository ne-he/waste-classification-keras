"""Data loading and preprocessing for waste image classification.

Wraps Keras' :class:`ImageDataGenerator` to provide deterministic train/val
splits, configurable augmentation driven by :class:`~src.config.DataConfig`,
and dataset integrity validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from src.config import DataConfig
from src.exceptions import DataError
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Loads and preprocesses the TrashNet dataset.

    Provides train/validation generators with configurable augmentation and a
    deterministic split. Generators are built lazily and cached on first use.

    Attributes:
        EXPECTED_CLASSES: The six waste categories the dataset must contain.
        MIN_IMAGES_PER_CLASS: Minimum images required per class to be valid.
    """

    EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    MIN_IMAGES_PER_CLASS = 10

    def __init__(self, config: DataConfig) -> None:
        """Initialize the loader.

        Args:
            config: Data configuration (paths, image size, split, augmentation).
        """
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self._train_gen: Optional[DirectoryIterator] = None
        self._val_gen: Optional[DirectoryIterator] = None

    def validate(self) -> None:
        """Validate dataset structure on disk.

        Checks that the dataset directory exists, contains exactly the expected
        class folders, and that each class has at least
        :attr:`MIN_IMAGES_PER_CLASS` images.

        Raises:
            DataError: If any structural check fails.
        """
        if not self.dataset_path.exists():
            logger.error("Dataset path does not exist: %s", self.dataset_path)
            raise DataError(f"Dataset path does not exist: {self.dataset_path}")

        actual_classes = sorted(d.name for d in self.dataset_path.iterdir() if d.is_dir())
        if set(actual_classes) != set(self.EXPECTED_CLASSES):
            logger.error(
                "Class mismatch: expected %s, found %s", self.EXPECTED_CLASSES, actual_classes
            )
            raise DataError(f"Expected classes {self.EXPECTED_CLASSES}, found {actual_classes}")

        for cls in actual_classes:
            count = len(list((self.dataset_path / cls).glob("*.jpg")))
            if count < self.MIN_IMAGES_PER_CLASS:
                logger.error("Class '%s' has only %d images", cls, count)
                raise DataError(
                    f"Class '{cls}' has only {count} images "
                    f"(minimum {self.MIN_IMAGES_PER_CLASS})"
                )
            logger.debug("  %s: %d images", cls, count)

        logger.info("Dataset validated: %d classes at %s", len(actual_classes), self.dataset_path)

    def get_generators(self) -> tuple[DirectoryIterator, DirectoryIterator]:
        """Build (or return cached) train and validation generators.

        The training generator applies augmentation; the validation generator
        only rescales. Both share a deterministic seed and split.

        Returns:
            A tuple ``(train_generator, val_generator)``.

        Raises:
            DataError: If the underlying Keras loader fails.
        """
        if self._train_gen is not None and self._val_gen is not None:
            return self._train_gen, self._val_gen

        aug = self.config.augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=self.config.val_split,
            rotation_range=aug.rotation_range,
            zoom_range=aug.zoom_range,
            horizontal_flip=aug.horizontal_flip,
            width_shift_range=aug.width_shift_range,
            height_shift_range=aug.height_shift_range,
        )
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=self.config.val_split,
        )

        common = dict(
            directory=str(self.dataset_path),
            target_size=(self.config.img_size, self.config.img_size),
            batch_size=self.config.batch_size,
            class_mode="categorical",
            seed=self.config.seed,
        )

        try:
            self._train_gen = train_datagen.flow_from_directory(subset="training", **common)
            self._val_gen = val_datagen.flow_from_directory(
                subset="validation", shuffle=False, **common
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to build data generators: %s", exc)
            raise DataError(f"Could not build data generators: {exc}") from exc

        logger.info(
            "Generators ready — train: %d samples | val: %d samples",
            self._train_gen.samples,
            self._val_gen.samples,
        )
        return self._train_gen, self._val_gen

    def get_class_labels(self) -> list[str]:
        """Return class names ordered by their integer index.

        Returns:
            Class names sorted by the Keras ``class_indices`` mapping.
        """
        if self._train_gen is None:
            self.get_generators()
        assert self._train_gen is not None  # for type checkers
        return list(self._train_gen.class_indices.keys())

    def get_class_distribution(self) -> dict[str, int]:
        """Count images per class directly from disk.

        Returns:
            A mapping of class name to image count.
        """
        return {
            cls.name: len(list(cls.glob("*.jpg")))
            for cls in sorted(self.dataset_path.iterdir())
            if cls.is_dir()
        }

    def get_y_train(self) -> np.ndarray:
        """Return integer training labels (for class-weight computation).

        Returns:
            A 1-D array of integer class indices for the training split.
        """
        if self._train_gen is None:
            self.get_generators()
        assert self._train_gen is not None  # for type checkers
        return self._train_gen.classes
