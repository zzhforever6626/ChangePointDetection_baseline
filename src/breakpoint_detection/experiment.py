from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import select_peaks, sequential_tolerance_metrics, tolerance_metrics


@dataclass(frozen=True)
class TemporalSplit:
    """Contiguous train/validation/test bounds for one time series."""

    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


def yahoo_a4_split(length: int) -> TemporalSplit:
    """Return the 40/20/40 temporal split reported for Yahoo A4."""
    if length < 5:
        raise ValueError("Yahoo A4 paper split requires at least five samples")
    train_stop = int(np.floor(0.4 * length))
    validation_stop = int(np.floor(0.6 * length))
    return TemporalSplit(
        train=(0, train_stop),
        validation=(train_stop, validation_stop),
        test=(validation_stop, length),
    )


def split_labels(length: int, split: TemporalSplit) -> np.ndarray:
    labels = np.full(length, "test", dtype=object)
    labels[slice(*split.train)] = "train"
    labels[slice(*split.validation)] = "validation"
    return labels


def candidate_peak_scores(scores: np.ndarray) -> np.ndarray:
    """Keep finite local maxima while preserving their absolute indices."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    candidates = np.full(len(scores), np.nan)
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return candidates
    finite_scores = scores[finite]
    left = np.r_[-np.inf, finite_scores[:-1]]
    right = np.r_[finite_scores[1:], -np.inf]
    peaks = (finite_scores > left) & (finite_scores >= right)
    candidates[finite[peaks]] = finite_scores[peaks]
    return candidates


def threshold_predictions(
    candidate_scores: np.ndarray,
    threshold: float,
    bounds: tuple[int, int],
    min_distance: int,
) -> np.ndarray:
    """Threshold candidate peaks inside a split and apply score-priority NMS."""
    start, stop = bounds
    scores = np.asarray(candidate_scores, dtype=float).reshape(-1)
    selected = np.full(len(scores), np.nan)
    region = scores[start:stop]
    keep = np.isfinite(region) & (region >= threshold)
    selected[start:stop][keep] = region[keep]
    return select_peaks(selected, int(keep.sum()), min_distance=min_distance)


def aggregate_tolerance_metrics(
    truths: Iterable[np.ndarray], predictions: Iterable[np.ndarray], margin: int,
    *, strict: bool = False,
) -> dict[str, float | int]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    errors: list[float] = []
    for truth, prediction in zip(truths, predictions, strict=True):
        row = tolerance_metrics(truth, prediction, margin, strict=strict)
        for key in totals:
            totals[key] += int(row[key])
        if np.isfinite(row["mean_delay"]):
            errors.extend([float(row["mean_delay"])] * int(row["tp"]))
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": float(np.mean(errors)) if errors else float("nan"),
    }


def aggregate_sequential_metrics(
    truths: Iterable[np.ndarray],
    predictions: Iterable[np.ndarray],
    bounds: Iterable[tuple[int, int]],
    margin: int,
) -> dict[str, float | int]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for truth, prediction, region in zip(truths, predictions, bounds, strict=True):
        row = sequential_tolerance_metrics(truth, prediction, margin, region)
        for key in totals:
            totals[key] += int(row[key])
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**totals, "precision": precision, "recall": recall, "f1": f1}
