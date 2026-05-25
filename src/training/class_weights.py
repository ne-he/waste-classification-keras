"""Class-weight computation for handling dataset imbalance.

The TrashNet dataset is heavily imbalanced (``trash`` has only 137 images
versus ~500 for other classes). Passing balanced class weights to
``model.fit`` penalizes errors on rare classes more heavily, which is the key
lever for improving the ``trash`` F1-score.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.exceptions import TrainingError
from src.logger import setup_logger

logger = setup_logger(__name__)


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights from integer training labels.

    Uses scikit-learn's ``'balanced'`` heuristic, where each class weight is
    ``n_samples / (n_classes * n_samples_for_class)``.

    Args:
        labels: 1-D array of integer class indices for the training set.

    Returns:
        A mapping of class index to weight, suitable for the ``class_weight``
        argument of ``keras.Model.fit``.

    Raises:
        TrainingError: If ``labels`` is empty.

    Example:
        >>> import numpy as np
        >>> compute_class_weights(np.array([0, 0, 0, 1]))
        {0: 0.666..., 1: 2.0}
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        logger.error("Cannot compute class weights from empty label array.")
        raise TrainingError("Cannot compute class weights: label array is empty.")

    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_map = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    logger.info("Computed class weights: %s", weight_map)
    return weight_map
