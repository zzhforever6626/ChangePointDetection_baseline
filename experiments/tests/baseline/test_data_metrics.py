import numpy as np
import pandas as pd

from breakpoint_detection.data import load_yahoo_csv
from breakpoint_detection.metrics import label_indices, tolerance_metrics


def test_yahoo_loader_and_event_conversion(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({
        "timestamp": [1, 2, 3, 4],
        "value": [1.0, None, 3.0, 4.0],
        "is_anomaly": [0, 1, 1, 0],
    }).to_csv(path, index=False)
    series = load_yahoo_csv(path)
    np.testing.assert_allclose(series.values, [1, 2, 3, 4])
    np.testing.assert_array_equal(label_indices(series.labels), [1])


def test_tolerance_metrics_are_one_to_one():
    metrics = tolerance_metrics(np.array([10, 30]), np.array([9, 11, 40]), tolerance=2)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 2
    assert metrics["fn"] == 1


def test_tolerance_matching_maximizes_true_positives():
    metrics = tolerance_metrics(np.array([0, 3]), np.array([2, 4]), tolerance=2)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 0, 0)


def test_yahoo_a4_auto_uses_changepoint_not_anomaly(tmp_path):
    path = tmp_path / "A4Benchmark-TS1.csv"
    pd.DataFrame({
        "timestamps": [1, 2, 3, 4],
        "value": [2.0, 3.0, 7.0, 8.0],
        "anomaly": [0, 1, 0, 0],
        "changepoint": [0, 0, 1, 0],
    }).to_csv(path, index=False)
    series = load_yahoo_csv(path)
    assert series.label_column == "changepoint"
    np.testing.assert_array_equal(label_indices(series.labels), [2])


def test_tscp2_a4_preprocessing_replaces_spatial_anomaly(tmp_path):
    path = tmp_path / "A4Benchmark-TS1.csv"
    pd.DataFrame({
        "value": [0.0, 100.0, 4.0, 6.0],
        "anomaly": [0, 1, 0, 0],
        "changepoint": [0, 0, 1, 0],
    }).to_csv(path, index=False)
    series = load_yahoo_csv(path, preprocessing="tscp2-a4")
    np.testing.assert_allclose(series.values, [0.0, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_array_equal(label_indices(series.labels), [2])
