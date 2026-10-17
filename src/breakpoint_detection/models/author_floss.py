"""Literal numerical port of the supplied author's streaming FLOSS example.

The source is figure_16/figure_13/{stamp1DirSelf,stamp1DirSelfSlide,
getAsymIAC,getArcCurve,pick_k_minimum,pammap_dataset}.m, preserved in the
provided-source-comparison-2026-10-06 experiment. No STUMPY code is used.

The matrix-profile indices deliberately remain MATLAB one-based: shifting
them by -1 is part of stamp1DirSelfSlide, rather than an API coordinate
conversion. Public event indices from pick_k_minimum are Python zero-based.
The driver's literal crc_final(mid_idx+i)=crc(mid_idx) is also retained:
it samples local index mid_idx-1 but reports global index start+mid_idx.

An explicit NumPy Generator makes the single random asymmetric IAC
repeatable. Its discrete distribution agrees with randi's inclusive
rightward sampling (including self matches); MATLAB and NumPy random
streams are not bitwise interchangeable. There is no IAC averaging,
edge exclusion, score threshold, event history, or end-coordinate shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def matlab_round_positive(value: float) -> int:
    """MATLAB round for the nonnegative lengths used by the example."""
    return int(np.floor(value + 0.5))


def _series(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("The time series must be one dimensional")
    return result


def _check_input(values: NDArray[np.float64], m: int) -> None:
    if int(m) != m:
        raise ValueError("Subsequence length must be an integer")
    if m > len(values) / 2:
        raise ValueError("Time series is too short relative to subsequence length")
    if m < 4:
        raise ValueError("Subsequence length must be at least 4")


@dataclass(frozen=True)
class _MassPre:
    spectrum: NDArray[np.complex128]
    n: int
    sumx2: NDArray[np.float64]
    sumx: NDArray[np.float64]
    meanx: NDArray[np.float64]
    sigmax2: NDArray[np.float64]
    sigmax: NDArray[Any]


def _fast_find_nn_pre(x: NDArray[np.float64], m: int) -> _MassPre:
    """Translate stamp1DirSelf.m's fastfindNNPre, including its cumsums."""
    n = len(x)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = x
    spectrum = np.fft.fft(padded)
    cum_sumx = np.cumsum(padded)
    cum_sumx2 = np.cumsum(padded**2)
    sumx2 = cum_sumx2[m - 1 : n] - np.concatenate(([0.0], cum_sumx2[: n - m]))
    sumx = cum_sumx[m - 1 : n] - np.concatenate(([0.0], cum_sumx[: n - m]))
    meanx = sumx / m
    sigmax2 = sumx2 / m - meanx**2
    # MATLAB sqrt can produce a complex value for negative roundoff. Do not
    # clamp variances or replace constant windows with a special distance.
    sigmax = np.emath.sqrt(sigmax2)
    return _MassPre(spectrum, n, sumx2, sumx, meanx, sigmax2, sigmax)


def _fast_find_nn_squared(pre: _MassPre, query: NDArray[np.float64]) -> NDArray[Any]:
    """Translate the FFT MASS expression before the source's sqrt/real."""
    m = len(query)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized = (query - np.mean(query)) / np.std(query, ddof=0)
        padded = np.zeros(2 * pre.n, dtype=np.float64)
        padded[:m] = normalized[::-1]
        z = np.fft.ifft(pre.spectrum * np.fft.fft(padded))
        sumy = np.sum(padded)
        sumy2 = np.sum(padded**2)
        return (
            (pre.sumx2 - 2 * pre.sumx * pre.meanx + m * pre.meanx**2) / pre.sigmax2
            - 2 * (z[m - 1 : pre.n] - sumy * pre.meanx) / pre.sigmax
            + sumy2
        )


def stamp_one_directional_self(
    values: ArrayLike, m: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port stamp1DirSelf: rightward nearest neighbors, one-based indices.

    Query indices descend. Squared real distances are compared and stored,
    and sqrt/real is performed only after all queries, exactly as in MATLAB.
    Neighbors at distance exactly round(m/2) are excluded.
    """
    x = _series(values)
    _check_input(x, m)
    m = int(m)
    pro_len = len(x) - m + 1
    radius = matlab_round_positive(m / 2)
    matrix_profile = np.full(pro_len, np.inf)
    profile_index = np.full(pro_len, np.inf)
    pre = _fast_find_nn_pre(x, m)
    for idx in range(pro_len, 0, -1):
        query = x[idx - 1 : idx + m - 1]
        distance = np.real(_fast_find_nn_squared(pre, query))
        exclusion_start = max(1, idx - radius)
        distance[exclusion_start - 1 :] = np.inf
        update = distance < matrix_profile
        profile_index[update] = idx
        matrix_profile[update] = distance[update]
    with np.errstate(invalid="ignore"):
        matrix_profile = np.real(np.emath.sqrt(matrix_profile))
    return matrix_profile, profile_index


def stamp_one_directional_self_slide(
    values: ArrayLike,
    m: int,
    matrix_profile: ArrayLike,
    profile_index: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port stamp1DirSelfSlide for the already-shifted, fixed-length data.

    Existing Euclidean distances are shifted, index values are decremented,
    and only the newest query is compared. No full self-join is recomputed.
    """
    x = _series(values)
    _check_input(x, m)
    m = int(m)
    pro_len = len(x) - m + 1
    p = np.asarray(matrix_profile, dtype=np.float64).copy()
    indices = np.asarray(profile_index, dtype=np.float64).copy()
    if p.shape != (pro_len,) or indices.shape != (pro_len,):
        raise ValueError("Profile arrays must have length len(values)-m+1")
    p[:-1] = p[1:]
    indices[:-1] = indices[1:]
    p[-1] = np.inf
    indices[-1] = np.inf
    indices -= 1
    squared = _fast_find_nn_squared(_fast_find_nn_pre(x, m), x[-m:])
    # The slide source applies real after sqrt, not before it.
    with np.errstate(invalid="ignore"):
        distance = np.real(np.sqrt(squared.astype(np.complex128)))
    exclusion_start = pro_len - matlab_round_positive(m / 2)
    distance[exclusion_start - 1 :] = np.inf
    update = distance < p
    indices[update] = pro_len
    p[update] = distance[update]
    return p, indices


def get_asym_iac(
    pro_len: int, rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Port getAsymIAC: one random inclusive rightward link per index."""
    if pro_len <= 0:
        raise ValueError("Profile length must be positive")
    if rng is None:
        rng = np.random.default_rng(0)
    links = np.empty(pro_len, dtype=np.float64)
    for index in range(pro_len):
        links[index] = rng.integers(index + 1, pro_len + 1)
    return get_arc_curve(links, pro_len)


def get_arc_curve(profile_index: ArrayLike, t_len: int) -> NDArray[np.float64]:
    """Port getArcCurve, accepting MATLAB one-based nearest neighbors.

    Each link contributes on [min(i,j), max(i,j)) in one-based coordinates.
    Invalid links are errors as in MATLAB indexing, rather than dropped.
    """
    indices = np.asarray(profile_index, dtype=np.float64)
    if indices.ndim != 1:
        raise ValueError("Profile indices must be one dimensional")
    if len(indices) > t_len:
        raise ValueError("There are more source links than arc-curve positions")
    if np.any(~np.isfinite(indices)) or np.any(indices != np.floor(indices)):
        raise ValueError("The source arc-curve routine cannot index nonfinite/noninteger neighbors")
    if np.any(indices < 1) or np.any(indices > t_len):
        raise ValueError("The source arc-curve routine cannot index out-of-range neighbors")
    nnmark = np.zeros(t_len, dtype=np.float64)
    source = np.arange(1, len(indices) + 1, dtype=np.int64)
    destination = indices.astype(np.int64)
    # Repeated positions must accumulate. add.at reproduces the scalar loop.
    np.add.at(nnmark, np.minimum(source, destination) - 1, 1.0)
    np.add.at(nnmark, np.maximum(source, destination) - 1, -1.0)
    return np.cumsum(nnmark)


def corrected_arc_curve(arc: ArrayLike, iac: ArrayLike) -> NDArray[np.float64]:
    """Port pammap_dataset's crc=arc./iac; crc=min(1,crc).

    MATLAB's documented default minimum omits NaN values, including for
    min(A,B), so the source maps a 0/0 entry to 1 in this operation. This
    does not repair the NaN distances caused by constant subsequences.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.fmin(1.0, np.asarray(arc) / np.asarray(iac))


def pick_k_minimum(curve: ArrayLike, k: int, step: int) -> NDArray[np.int64]:
    """Port pick_k_minimum with zero-based return indices and closed ±step.

    The source always makes k selections, including its first-index fallback
    when all scores are infinite. This behavior is deliberately retained.
    The documented MATLAB default omits NaN values, and minima choose the
    first tied position. An all-NaN vector retains the first-index fallback.
    """
    values = np.asarray(curve, dtype=np.float64).copy()
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Curve must be a nonempty one-dimensional vector")
    if k < 0 or int(k) != k or step < 0 or int(step) != step:
        raise ValueError("k and step must be nonnegative integers")
    chosen = np.zeros(int(k), dtype=np.int64)
    for i in range(int(k)):
        eligible = np.flatnonzero(~np.isnan(values))
        chosen[i] = eligible[np.argmin(values[eligible])] if len(eligible) else 0
        lo = max(0, int(chosen[i]) - int(step))
        hi = min(len(values) - 1, int(chosen[i]) + int(step))
        values[lo : hi + 1] = np.inf
    return chosen


class AuthorStreamingFloss:
    """Stateful port of the source driver's MP update and center sampling.

    Initial construction consumes the complete supplied initial buffer.
    update(value) then advances one sample. Scores remain center-curve
    samples; event selection is the source's offline top-k final operation.
    """

    def __init__(
        self,
        initial_buffer: ArrayLike,
        m: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.buffer_ = _series(initial_buffer).copy()
        _check_input(self.buffer_, m)
        self.m = int(m)
        self.buffer_size_ = len(self.buffer_)
        self.pro_len_ = self.buffer_size_ - self.m + 1
        self.exclusion_radius_ = matlab_round_positive(self.m / 2)
        self.mid_idx_ = matlab_round_positive(self.buffer_size_ / 2)
        self.buffer_start_ = 0
        self.iac_ = get_asym_iac(self.pro_len_, rng)
        self.P_, self.I_ = stamp_one_directional_self(self.buffer_, self.m)
        self._refresh_center()

    def _refresh_center(self) -> None:
        # pammap_dataset passes pro_idx(1:end-exc_zone-1): only the last
        # radius+1 entries are truncated, with no added edge masking.
        useful = self.I_[: self.pro_len_ - self.exclusion_radius_ - 1]
        self.arc_ = get_arc_curve(useful, self.pro_len_)
        self.crc_ = corrected_arc_curve(self.arc_, self.iac_)
        self.center_arc_ = float(self.arc_[self.mid_idx_ - 1])
        self.center_score_ = float(self.crc_[self.mid_idx_ - 1])
        self.center_report_position_ = self.center_report_position(self.buffer_start_)
        self.center_available_stop_ = self.buffer_start_ + self.buffer_size_

    def update(self, value: float) -> float:
        self.buffer_[:-1] = self.buffer_[1:]
        self.buffer_[-1] = float(value)
        self.P_, self.I_ = stamp_one_directional_self_slide(
            self.buffer_, self.m, self.P_, self.I_
        )
        self.buffer_start_ += 1
        self._refresh_center()
        return self.center_score_

    def center_score(self) -> float:
        return self.center_score_

    def center_arc_count(self) -> float:
        return self.center_arc_

    def center_report_position(self, buffer_start: int | None = None) -> int:
        if buffer_start is None:
            buffer_start = self.buffer_start_
        return int(buffer_start) + self.mid_idx_

    @property
    def available_stop(self) -> int:
        """Exclusive consumed-data stop for the current center sample."""
        return self.center_available_stop_


# Source-name aliases make direct source-to-port comparisons easier to read.
stamp1DirSelf = stamp_one_directional_self
stamp1DirSelfSlide = stamp_one_directional_self_slide
getAsymIAC = get_asym_iac
getArcCurve = get_arc_curve
