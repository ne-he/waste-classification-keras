"""Model factory for waste image classification.

Provides two architectures refactored from the original notebook — a baseline
CNN trained from scratch and a MobileNetV2 transfer-learning model — plus a
dispatching factory, a compilation helper, and a Grad-CAM layer locator.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import ModelConfig, TrainingConfig
from src.exceptions import ModelError
from src.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_IMG_SIZE = 224


def build_baseline_cnn(config: ModelConfig, img_size: int = DEFAULT_IMG_SIZE) -> keras.Model:
    """Build the baseline CNN trained from scratch.

    Architecture (matches the original notebook): three Conv2D + MaxPooling2D
    blocks with 32/64/128 filters, then Flatten, Dense(128), Dropout, and a
    softmax classification head.

    Args:
        config: Model configuration (number of classes, dropout rate).
        img_size: Input image height/width in pixels.

    Returns:
        An uncompiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(img_size, img_size, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.num_classes, activation="softmax"),
        ],
        name="Baseline_CNN",
    )
    logger.info("Built baseline CNN (%d params)", model.count_params())
    return model


def build_mobilenet_v2(config: ModelConfig, img_size: int = DEFAULT_IMG_SIZE) -> keras.Model:
    """Build the MobileNetV2 transfer-learning model.

    A MobileNetV2 backbone (ImageNet weights when ``config.pretrained``) is
    topped with GlobalAveragePooling2D, Dense(128), Dropout, and a softmax head.
    The backbone is frozen when ``config.freeze_backbone`` is set.

    Args:
        config: Model configuration (classes, dropout, pretrained, freeze).
        img_size: Input image height/width in pixels.

    Returns:
        An uncompiled Keras model.
    """
    weights = "imagenet" if config.pretrained else None
    base_model = keras.applications.MobileNetV2(
        weights=weights,
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = not config.freeze_backbone

    model = keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.num_classes, activation="softmax"),
        ],
        name="MobileNetV2_Transfer",
    )
    logger.info(
        "Built MobileNetV2 (pretrained=%s, frozen=%s, %d params)",
        config.pretrained,
        config.freeze_backbone,
        model.count_params(),
    )
    return model


def build_model(config: ModelConfig, img_size: int = DEFAULT_IMG_SIZE) -> keras.Model:
    """Build a model by dispatching on ``config.architecture``.

    Args:
        config: Model configuration; ``architecture`` selects the builder.
        img_size: Input image height/width in pixels.

    Returns:
        An uncompiled Keras model.

    Raises:
        ModelError: If ``config.architecture`` is not supported.

    Example:
        >>> from src.config import ModelConfig
        >>> model = build_model(ModelConfig(architecture="mobilenet_v2"))
    """
    builders = {
        "baseline_cnn": build_baseline_cnn,
        "mobilenet_v2": build_mobilenet_v2,
    }
    builder = builders.get(config.architecture)
    if builder is None:
        logger.error("Unknown architecture requested: %s", config.architecture)
        raise ModelError(
            f"Unknown architecture '{config.architecture}'. " f"Supported: {sorted(builders)}"
        )
    return builder(config, img_size)


def compile_model(model: keras.Model, config: TrainingConfig) -> keras.Model:
    """Compile a model for categorical classification.

    Args:
        model: The Keras model to compile.
        config: Training configuration providing optimizer and learning rate.

    Returns:
        The same model, compiled in place.

    Raises:
        ModelError: If the optimizer name is not supported.
    """
    optimizers = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }
    optimizer_cls = optimizers.get(config.optimizer.lower())
    if optimizer_cls is None:
        logger.error("Unknown optimizer: %s", config.optimizer)
        raise ModelError(f"Unknown optimizer '{config.optimizer}'. Supported: {sorted(optimizers)}")

    model.compile(
        optimizer=optimizer_cls(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Compiled model with %s (lr=%g)", config.optimizer, config.learning_rate)
    return model


def get_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last Conv2D layer, searching nested models.

    Used by Grad-CAM. Handles models whose backbone is itself a nested
    functional model (e.g. MobileNetV2 wrapped in a Sequential).

    Args:
        model: The Keras model to inspect.

    Returns:
        The name of the last Conv2D layer found.

    Raises:
        ModelError: If no Conv2D layer exists in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, layers.Conv2D):
                    return sub_layer.name
    logger.error("No Conv2D layer found in model '%s'", model.name)
    raise ModelError("No Conv2D layer found in model")
