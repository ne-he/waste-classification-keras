"""Model evaluation: metrics, confusion matrix, per-class AUC, and reports.

Refactored from the notebook's evaluation cells. The :class:`Evaluator` runs
predictions once and caches them, then derives all metrics and artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")  # headless backend for report generation
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.data import DataLoader  # noqa: E402
from src.exceptions import ModelError  # noqa: E402
from src.logger import setup_logger  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402

logger = setup_logger(__name__)

PathLike = Union[str, Path]


class Evaluator:
    """Computes evaluation metrics and reports for a trained model.

    Args:
        model: A trained Keras model.
        data_loader: Data loader providing the validation generator.
        class_names: Ordered class names matching the model's output indices.
    """

    def __init__(self, model: keras.Model, data_loader: DataLoader, class_names: list[str]) -> None:
        self.model = model
        self.data_loader = data_loader
        self.class_names = class_names
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._y_prob: Optional[np.ndarray] = None

    def _ensure_predictions(self) -> None:
        """Run predictions on the validation set once and cache the results.

        Raises:
            ModelError: If prediction fails.
        """
        if self._y_prob is not None:
            return
        _, val_gen = self.data_loader.get_generators()
        try:
            val_gen.reset()
            self._y_prob = self.model.predict(val_gen, verbose=0)
        except (RuntimeError, ValueError) as exc:
            logger.error("Prediction failed: %s", exc)
            raise ModelError(f"Prediction failed: {exc}") from exc
        self._y_pred = np.argmax(self._y_prob, axis=1)
        self._y_true = np.asarray(val_gen.classes)
        logger.info("Cached predictions for %d validation samples", self._y_true.shape[0])

    def evaluate(self) -> dict:
        """Compute the full set of evaluation metrics.

        Returns:
            A dict with ``accuracy``, ``macro_auc``, ``per_class`` (precision/
            recall/f1/support per class), and ``confusion_matrix``.
        """
        self._ensure_predictions()
        assert self._y_true is not None and self._y_pred is not None
        report = self.generate_classification_report()
        cm = self.generate_confusion_matrix()
        results = {
            "accuracy": float(report["accuracy"]),
            "macro_auc": self.compute_per_class_auc()["macro"],
            "per_class": {name: report[name] for name in self.class_names if name in report},
            "confusion_matrix": cm.tolist(),
        }
        logger.info("Evaluation accuracy: %.4f", results["accuracy"])
        return results

    def generate_classification_report(self) -> dict:
        """Return the sklearn classification report as a dict.

        Returns:
            Per-class and aggregate precision/recall/F1/support.
        """
        self._ensure_predictions()
        return classification_report(
            self._y_true,
            self._y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

    def generate_confusion_matrix(self, save_path: Optional[PathLike] = None) -> np.ndarray:
        """Compute the confusion matrix, optionally saving a heatmap plot.

        Args:
            save_path: If given, a PNG heatmap is written to this path.

        Returns:
            The confusion matrix as a 2-D integer array.
        """
        self._ensure_predictions()
        cm = confusion_matrix(self._y_true, self._y_pred)
        if save_path is not None:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
            )
            ax.set_title("Confusion Matrix (Validation Set)")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            fig.tight_layout()
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
            logger.info("Saved confusion matrix to %s", save_path)
        return cm

    def compute_per_class_auc(self) -> dict[str, float]:
        """Compute one-vs-rest ROC AUC per class plus the macro average.

        Returns:
            A mapping of class name to AUC, including a ``"macro"`` key.
        """
        self._ensure_predictions()
        assert self._y_true is not None and self._y_prob is not None
        n_classes = len(self.class_names)
        true_bin = label_binarize(self._y_true, classes=list(range(n_classes)))

        aucs: dict[str, float] = {}
        fpr_list, tpr_list = [], []
        for i, name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(true_bin[:, i], self._y_prob[:, i])
            aucs[name] = float(auc(fpr, tpr))
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in zip(fpr_list, tpr_list):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        aucs["macro"] = float(auc(all_fpr, mean_tpr))
        return aucs

    def generate_full_report(self, output_dir: PathLike) -> Path:
        """Generate a Markdown evaluation report with a confusion-matrix plot.

        Args:
            output_dir: Directory where the report and plot are written.

        Returns:
            Path to the generated Markdown report file.
        """
        output_dir = ensure_dir(output_dir)
        results = self.evaluate()
        aucs = self.compute_per_class_auc()
        self.generate_confusion_matrix(output_dir / "confusion_matrix.png")
        save_json(results, output_dir / "metrics.json")

        report_path = output_dir / "evaluation_report.md"
        lines = [
            "# Evaluation Report",
            "",
            f"- **Accuracy:** {results['accuracy']:.4f}",
            f"- **Macro AUC:** {results['macro_auc']:.4f}",
            "",
            "## Per-Class Metrics",
            "",
            "| Class | Precision | Recall | F1-Score | AUC | Support |",
            "|-------|-----------|--------|----------|-----|---------|",
        ]
        for name in self.class_names:
            metrics = results["per_class"].get(name, {})
            lines.append(
                f"| {name} | {metrics.get('precision', 0):.4f} | "
                f"{metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | "
                f"{aucs.get(name, 0):.4f} | {int(metrics.get('support', 0))} |"
            )
        lines += ["", "![Confusion Matrix](confusion_matrix.png)", ""]

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved evaluation report to %s", report_path)
        return report_path
