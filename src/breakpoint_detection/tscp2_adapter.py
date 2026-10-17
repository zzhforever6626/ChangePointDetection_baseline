"""Label-free TS-CP2 sampling and detection, with disclosed Yahoo adapters."""
import numpy as np


def training_pairs(values, train_stops, window, stride=5, last_start_inclusive=True):
    """Adjacent windows wholly inside each training prefix; never concatenate series."""
    pairs, locations = [], []
    for series, (row, stop) in enumerate(zip(values, train_stops)):
        for start in range(0, int(stop) - 2 * window + int(last_start_inclusive), stride):
            pairs.append(row[start:start + 2 * window])
            locations.append((series, start))
    if not pairs:
        raise ValueError("no complete training pairs for the registered window")
    return np.asarray(pairs, dtype=np.float32), np.asarray(locations, dtype=np.int64)


def separated_batches(locations, batch_size, distance, seed):
    """Shuffle once as the source; enforce paper separation within the same series.

    The paper leaves the distance application dependent. Here distance=2W makes
    same-series positive-pair supports disjoint. Cross-series pairs are eligible.
    Incomplete final batches are dropped, as in the author's training loader.
    """
    remaining = np.random.RandomState(seed).permutation(len(locations)).tolist()
    batches = []
    while len(remaining) >= batch_size:
        batch, deferred, chosen = [], [], {}
        for index in remaining:
            series, start = map(int, locations[index])
            if len(batch) < batch_size and all(abs(start - other) >= distance
                                               for other in chosen.get(series, [])):
                batch.append(index)
                chosen.setdefault(series, []).append(start)
            else:
                deferred.append(index)
        if len(batch) < batch_size:
            break
        batches.append(batch)
        remaining = deferred
    if not batches:
        raise ValueError("too few separated pairs for a complete training batch")
    return np.asarray(batches, dtype=np.int64)


def similarity_drop(similarity, window, step=1):
    """2021 Figure 7: mean of PREVIOUS W similarities minus current similarity."""
    similarity = np.asarray(similarity, dtype=float)
    result = np.full(similarity.shape, np.nan)
    positions = np.flatnonzero(np.isfinite(similarity))
    prefix = np.r_[0., np.cumsum(similarity[positions])]
    for index in range(window, len(positions)):
        if np.all(np.diff(positions[index-window:index+1]) == step):
            center = positions[index]
            result[center] = (prefix[index] - prefix[index-window]) / window - similarity[center]
    return result


def phase_peak_scores(scores, window, phase_stop, step=1):
    """Local maxima require both neighbors and cannot use a future phase input.

    Leftmost point represents a flat peak. No endpoint maxima or extra NMS.
    Reading the right neighbor requires observations through center+W+1.
    """
    scores = np.asarray(scores, dtype=float)
    peaks = np.full(scores.shape, np.nan)
    for center in range(step, min(len(scores)-step, int(phase_stop)-window-step+1)):
        left, value, right = scores[[center-step,center,center+step]]
        if np.isfinite([left, value, right]).all() and value > left and value >= right:
            peaks[center] = value
    return peaks
