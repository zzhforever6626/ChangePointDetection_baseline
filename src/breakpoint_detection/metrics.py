from __future__ import annotations

import numpy as np


def paper_event_metrics(truth, prediction, margin):
    """2021 event counts; overlapping margins use a closest-edge-first adapter.

    Equal-coordinate predictions remain separate; no intervals or FP runs merge.
    """
    if margin <= 0:
        raise ValueError('margin must be positive')
    truth = np.sort(np.asarray(truth, dtype=int))
    prediction = np.sort(np.asarray(prediction, dtype=int))
    edges = sorted((abs(int(p)-int(t)), int(p), int(t), i, j)
                   for i, p in enumerate(prediction) for j, t in enumerate(truth)
                   if abs(int(p)-int(t)) < margin)
    used_p, used_t, errors = set(), set(), []
    for distance, _, _, i, j in edges:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); errors.append(distance)
    tp = len(errors)
    fp, fn = len(prediction)-tp, len(truth)-tp
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp/len(prediction) if len(prediction) else 0.,
                recall=tp/len(truth) if len(truth) else 0.,
                f1=2*tp/(len(prediction)+len(truth)) if len(prediction)+len(truth) else 0.,
                mean_delay=float(np.mean(errors)) if errors else float('nan'))


def label_indices(labels: np.ndarray) -> np.ndarray:
    """Convert binary labels to event indices (one event per positive run)."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    return np.flatnonzero(labels & ~np.r_[False, labels[:-1]])


def select_peaks(scores: np.ndarray, count: int, min_distance: int = 1) -> np.ndarray:
    """Select the highest finite scores with non-maximum suppression."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if count <= 0:
        return np.empty(0, dtype=int)
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    chosen: list[int] = []
    for index in order:
        if not np.isfinite(scores[index]):
            continue
        if all(abs(int(index) - old) >= min_distance for old in chosen):
            chosen.append(int(index))
            if len(chosen) == count:
                break
    return np.asarray(sorted(chosen), dtype=int)


def tolerance_metrics(
    truth: np.ndarray, prediction: np.ndarray, tolerance: int, *, strict: bool = False
) -> dict[str, float | int]:
    """Maximize one-to-one matches, then minimize total timing error.

    The TS-CP2 Yahoo evaluation describes a hit as being *less than* its
    detection margin away; ``strict=True`` applies that boundary convention.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    truth = np.unique(np.asarray(truth, dtype=int))
    prediction = np.sort(np.asarray(prediction, dtype=int))
    # On a line, an optimal matching can be ordered. Dynamic programming avoids
    # the greedy pitfall where the closest pair consumes a prediction needed by
    # another truth and thereby lowers the number of true positives.
    n_pred, n_truth = len(prediction), len(truth)
    best = [[(0, 0) for _ in range(n_truth + 1)] for _ in range(n_pred + 1)]
    action = [["" for _ in range(n_truth + 1)] for _ in range(n_pred + 1)]
    for i in range(1, n_pred + 1):
        for j in range(1, n_truth + 1):
            best[i][j], action[i][j] = best[i - 1][j], "skip_prediction"
            if best[i][j - 1] > best[i][j]:
                best[i][j], action[i][j] = best[i][j - 1], "skip_truth"
            distance = abs(int(prediction[i - 1]) - int(truth[j - 1]))
            within_margin = distance < tolerance if strict else distance <= tolerance
            if within_margin:
                previous = best[i - 1][j - 1]
                candidate = (previous[0] + 1, previous[1] - distance)
                if candidate > best[i][j]:
                    best[i][j], action[i][j] = candidate, "match"
    errors: list[int] = []
    i, j = n_pred, n_truth
    while i and j:
        if action[i][j] == "match":
            errors.append(abs(int(prediction[i - 1]) - int(truth[j - 1])))
            i -= 1
            j -= 1
        elif action[i][j] == "skip_prediction":
            i -= 1
        else:
            j -= 1
    tp = len(errors)
    fp, fn = len(prediction) - tp, len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": float(np.mean(errors)) if errors else float("nan"),
    }


def sequential_tolerance_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    margin: int,
    bounds: tuple[int, int],
) -> dict[str, float | int]:
    """TS-CP2-style sequential F1 on intervals centered at each timestamp.

    A center is positive when it lies strictly inside the detection margin of
    a change point. Consecutive positive centers form one ground-truth event.
    Any predictions in that event count as one TP; remaining prediction runs
    count as false alarms. This reproduces the grouping rule, while the dense
    center grid is an explicit choice because the 2024 paper does not publish
    its exact interval sampling code.
    """
    start, stop = bounds
    if margin <= 0 or start < 0 or stop <= start:
        raise ValueError("margin must be positive and bounds must be nonempty")
    truth = np.unique(np.asarray(truth, dtype=int))
    prediction = np.unique(np.asarray(prediction, dtype=int))
    if np.any((truth < start) | (truth >= stop)):
        raise ValueError("truth must lie inside bounds")
    if np.any((prediction < start) | (prediction >= stop)):
        raise ValueError("predictions must lie inside bounds")

    # Build the union of positive intervals directly. The previous dense-mask
    # implementation had the same semantics but rebuilt arrays of length
    # ``stop - start`` for every validation threshold considered.
    intervals: list[tuple[int, int]] = []
    for point in truth:
        left = max(start, int(point) - margin + 1)
        right = min(stop, int(point) + margin)
        if intervals and left <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], right))
        else:
            intervals.append((left, right))

    covered = np.zeros(len(prediction), dtype=bool)
    tp = 0
    for left, right in intervals:
        first = int(np.searchsorted(prediction, left, side="left"))
        last = int(np.searchsorted(prediction, right, side="left"))
        if first < last:
            tp += 1
            covered[first:last] = True
    false_alarms = prediction[~covered]
    fp = (int(1 + np.count_nonzero(np.diff(false_alarms) > 1))
          if len(false_alarms) else 0)
    fn = len(intervals) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}
