"""Metrics: accuracy, per-class quality, calibration, and the robustness gap.

The headline number in this project is deliberately *not* i.i.d. test accuracy.  The shortcut
lives in the i.i.d. test split, so that number is high whether or not the model learned
anything about shapes.  What separates the two cases is the gap between the three test regimes,
and that gap is what :func:`robustness_report` computes.

Calibration is included because accuracy says nothing about whether a 0.99 output means 99%.
Expected calibration error is reported with its bin count attached, since ECE is not
comparable across binning schemes, and the per-bin reliability table is printed rather than
collapsed into one number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import CLASSES, Batch
from .train import batch_indices


@torch.no_grad()
def predict_logits(
    model: nn.Module, batch: Batch, device: str = "cpu", batch_size: int = 128
) -> np.ndarray:
    """Logits for a split, in eval mode and without augmentation."""
    was_training = model.training
    model.eval()
    model.to(device)
    images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
    chunks = []
    for index in batch_indices(images.shape[0], batch_size):
        chunks.append(model(images[index]).to("cpu"))
    if was_training:
        model.train()
    return torch.cat(chunks).numpy()


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if len(labels) == 0:
        raise ValueError("accuracy is undefined on an empty split")
    return float((np.asarray(predictions) == np.asarray(labels)).mean())


def confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray, n_classes: int = len(CLASSES)
) -> np.ndarray:
    """Rows are true classes, columns are predictions."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted in zip(np.asarray(labels), np.asarray(predictions)):
        matrix[int(true_label), int(predicted)] += 1
    return matrix


def per_class_report(
    predictions: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...] = CLASSES
) -> pd.DataFrame:
    """Precision, recall and F1 per class, with support.

    Zero-division is reported as 0.0 rather than dropped: a class the model never predicts is
    a finding, and silently omitting it flatters the macro average.
    """
    matrix = confusion_matrix(predictions, labels, len(class_names))
    rows = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        actual = int(matrix[index, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )
        rows.append(
            {
                "class": name,
                "support": actual,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    return pd.DataFrame(rows)


def macro_f1(
    predictions: np.ndarray, labels: np.ndarray, n_classes: int = len(CLASSES)
) -> float:
    """Unweighted mean F1 over classes, so a rare class cannot be ignored for free."""
    report = per_class_report(predictions, labels, CLASSES[:n_classes])
    return float(report["f1"].mean())


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Weighted mean gap between confidence and accuracy across confidence bins."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (N, n_classes)")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probabilities must sum to one per row")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == np.asarray(labels)).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        weight = in_bin.sum() / total
        error += weight * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def reliability_table(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Per-bin confidence against observed accuracy - the detail ECE averages away."""
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == np.asarray(labels)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        rows.append(
            {
                "bin": f"({low:.2f}, {high:.2f}]",
                "samples": count,
                "mean_confidence": round(float(confidence[in_bin].mean()), 4) if count else None,
                "accuracy": round(float(correct[in_bin].mean()), 4) if count else None,
                "gap": round(float(confidence[in_bin].mean() - correct[in_bin].mean()), 4)
                if count
                else None,
            }
        )
    return pd.DataFrame(rows)


def evaluate_split(
    model: nn.Module,
    batch: Batch,
    device: str = "cpu",
    batch_size: int = 128,
    bins: int = 10,
) -> dict[str, float]:
    """Everything worth reporting for one split, in one row."""
    logits = predict_logits(model, batch, device, batch_size)
    probabilities = softmax_probabilities(logits)
    predictions = probabilities.argmax(axis=1)
    return {
        "split": batch.name,
        "images": len(batch),
        "accuracy": round(accuracy(predictions, batch.labels), 4),
        "macro_f1": round(macro_f1(predictions, batch.labels), 4),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        f"ece@{bins}": round(expected_calibration_error(probabilities, batch.labels, bins), 4),
        "cue_agreement": round(batch.cue_agreement, 4),
        "cue_accuracy": round(
            float((predictions == batch.cue_labels).mean()), 4
        ),
    }


ROBUSTNESS_SPLITS: tuple[str, ...] = ("test", "test_cue_broken", "test_cue_inverted")


def robustness_report(
    model: nn.Module,
    datasets: dict[str, Batch],
    device: str = "cpu",
    splits: tuple[str, ...] = ROBUSTNESS_SPLITS,
    bins: int = 10,
) -> pd.DataFrame:
    """One row per test regime.

    ``cue_accuracy`` is the share of images where the prediction matches the *background tint*
    rather than the label.  On the inverted split those two questions have different answers,
    so a high ``cue_accuracy`` there is direct evidence that the model is reading the
    background - not an inference from an accuracy drop.
    """
    missing = [name for name in splits if name not in datasets]
    if missing:
        raise ValueError(f"missing splits: {missing}")
    rows = [evaluate_split(model, datasets[name], device, bins=bins) for name in splits]
    return pd.DataFrame(rows)


def shortcut_gap(report: pd.DataFrame) -> float:
    """Accuracy on the i.i.d. test set minus accuracy when the cue is inverted.

    Near zero means the model learned the object.  Large means the i.i.d. number was mostly
    the cue, and a single held-out score would never have shown it.
    """
    indexed = report.set_index("split")
    for required in ("test", "test_cue_inverted"):
        if required not in indexed.index:
            raise ValueError(f"the report is missing the {required!r} split")
    return round(
        float(indexed.loc["test", "accuracy"] - indexed.loc["test_cue_inverted", "accuracy"]), 4
    )
