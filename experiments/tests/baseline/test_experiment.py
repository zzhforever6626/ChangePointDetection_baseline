import numpy as np

from breakpoint_detection.experiment import (
    candidate_peak_scores,
    threshold_predictions,
    yahoo_a4_split,
)
from breakpoint_detection.metrics import sequential_tolerance_metrics, tolerance_metrics


def test_yahoo_a4_split_is_contiguous_40_20_40():
    split = yahoo_a4_split(10)
    assert split.train == (0, 4)
    assert split.validation == (4, 6)
    assert split.test == (6, 10)


def test_preselected_peaks_match_threshold_then_suppression():
    scores = np.full(30, np.nan)
    scores[[5, 8, 14, 17, 24]] = [2.0, 4.0, 1.0, 3.0, 4.0]
    for distance in (1, 5, 10):
        preselected = threshold_predictions(scores, -np.inf, (0, 30), distance)
        for threshold in (np.inf, 4.0, 3.0, 1.0, -np.inf):
            expected = threshold_predictions(scores, threshold, (0, 30), distance)
            actual = preselected[scores[preselected] >= threshold]
            np.testing.assert_array_equal(actual, expected)


def test_paper_margin_excludes_a_prediction_exactly_on_the_boundary():
    truth = np.array([100])
    assert tolerance_metrics(truth, np.array([124]), 24, strict=True)["tp"] == 0
    assert tolerance_metrics(truth, np.array([123]), 24, strict=True)["tp"] == 1
    assert tolerance_metrics(truth, np.array([124]), 24)["tp"] == 1


def test_sequential_f1_counts_one_hit_per_positive_interval_run():
    truth = np.array([100])
    prediction = np.array([85, 95, 105, 115])
    metrics = sequential_tolerance_metrics(truth, prediction, 10, (0, 200))
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 2, 0)
    assert sequential_tolerance_metrics(
        truth, np.array([110]), 10, (0, 200)
    )["tp"] == 0
