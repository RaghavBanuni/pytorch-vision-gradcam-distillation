"""Tests for the metrics, with every expected value computed by hand."""

from __future__ import annotations

import numpy as np
import pytest

from visionlab.evaluate import (
    accuracy,
    confusion_matrix,
    evaluate_split,
    expected_calibration_error,
    macro_f1,
    per_class_report,
    predict_logits,
    reliability_table,
    robustness_report,
    shortcut_gap,
    softmax_probabilities,
)

LABELS = np.array([0, 0, 1, 1])
PREDICTIONS = np.array([0, 1, 1, 1])


def test_accuracy_counts_matches():
    assert accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 2])) == pytest.approx(0.75)


def test_accuracy_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        accuracy(np.array([0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="empty split"):
        accuracy(np.array([]), np.array([]))


def test_confusion_matrix_places_truth_on_the_rows():
    matrix = confusion_matrix(PREDICTIONS, LABELS, n_classes=2)
    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert matrix.sum() == len(LABELS)


def test_per_class_report_is_hand_computable():
    report = per_class_report(PREDICTIONS, LABELS, class_names=("a", "b")).set_index("class")
    assert report.loc["a", "precision"] == pytest.approx(1.0)
    assert report.loc["a", "recall"] == pytest.approx(0.5)
    assert report.loc["a", "f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert report.loc["b", "precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert report.loc["b", "recall"] == pytest.approx(1.0)
    assert report.loc["b", "f1"] == pytest.approx(0.8)


def test_a_class_that_is_never_predicted_scores_zero_rather_than_disappearing():
    report = per_class_report(np.array([0, 0]), np.array([0, 1]), class_names=("a", "b"))
    row = report.set_index("class").loc["b"]
    assert row["support"] == 1
    assert row["precision"] == 0.0 and row["recall"] == 0.0 and row["f1"] == 0.0


def test_macro_f1_is_the_unweighted_mean():
    assert macro_f1(PREDICTIONS, LABELS, n_classes=2) == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-4)


def test_softmax_is_stable_and_normalised():
    probabilities = softmax_probabilities(np.array([[1000.0, 999.0], [0.0, 0.0]]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()
    assert probabilities[1].tolist() == [0.5, 0.5]


def test_expected_calibration_error_is_hand_computable():
    # both confidences land in the (0.5, 1.0] bin: accuracy 0.5, mean confidence 0.85
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels, bins=2) == pytest.approx(0.35)


def test_a_perfectly_calibrated_set_has_no_error():
    probabilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert expected_calibration_error(probabilities, np.array([0, 0]), bins=4) == pytest.approx(0.0)


def test_overconfidence_and_underconfidence_both_count():
    over = expected_calibration_error(np.array([[0.95, 0.05]]), np.array([1]), bins=10)
    under = expected_calibration_error(np.array([[0.55, 0.45]]), np.array([0]), bins=10)
    assert over == pytest.approx(0.95)
    assert under == pytest.approx(0.45)


def test_calibration_rejects_scores_that_are_not_probabilities():
    with pytest.raises(ValueError, match="sum to one"):
        expected_calibration_error(np.array([[2.0, 3.0]]), np.array([0]))
    with pytest.raises(ValueError, match="bins must be at least 2"):
        expected_calibration_error(np.array([[0.5, 0.5]]), np.array([0]), bins=1)


def test_reliability_table_accounts_for_every_sample():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45]])
    table = reliability_table(probabilities, np.array([0, 0, 1]), bins=5)
    assert len(table) == 5
    assert int(table["samples"].sum()) == 3
    populated = table.dropna(subset=["accuracy"])
    assert populated["mean_confidence"].between(0.0, 1.0).all()


def test_evaluate_split_reports_the_cue_as_well_as_the_label(trained, datasets):
    result = evaluate_split(trained[0], datasets["test"], bins=10)
    assert result["split"] == "test"
    assert result["images"] == len(datasets["test"])
    for key in ("accuracy", "macro_f1", "mean_confidence", "ece@10", "cue_accuracy"):
        assert 0.0 <= result[key] <= 1.0


def test_predict_logits_matches_the_split_size(trained, datasets):
    logits = predict_logits(trained[0], datasets["val"], batch_size=32)
    assert logits.shape == (len(datasets["val"]), 4)


def test_robustness_report_covers_the_three_regimes(trained, datasets):
    report = robustness_report(trained[0], datasets)
    assert list(report["split"]) == ["test", "test_cue_broken", "test_cue_inverted"]
    assert report["images"].nunique() == 1
    assert report.loc[report["split"] == "test_cue_inverted", "cue_agreement"].iloc[0] == 0.0


def test_the_shortcut_gap_is_the_difference_between_two_regimes(trained, datasets):
    report = robustness_report(trained[0], datasets)
    indexed = report.set_index("split")
    expected = indexed.loc["test", "accuracy"] - indexed.loc["test_cue_inverted", "accuracy"]
    assert shortcut_gap(report) == pytest.approx(expected, abs=1e-4)


def test_missing_splits_are_reported(trained, datasets):
    with pytest.raises(ValueError, match="missing splits"):
        robustness_report(trained[0], {"test": datasets["test"]})
    with pytest.raises(ValueError, match="missing the"):
        shortcut_gap(robustness_report(trained[0], datasets, splits=("test",)))
