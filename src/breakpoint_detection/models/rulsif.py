"""Formal RuLSIF wrapper around the frozen original-algorithm port.

The port follows the cited 2013 paper and MATLAB estimator; Yahoo n and lambda
follow TS-CP2 2021. Alarm mapping is an explicit adapter, not a recovered runner.
"""
from dataclasses import dataclass

import numpy as np

from ..experiment import candidate_peak_scores
from ..protocol import SETTINGS
from .author_rulsif import score_series


def chronological_alarm_scores(peaks, min_distance=None):
    """Registered Yahoo gap, compared to the preceding original peak.

Deduplication precedes thresholding. This is separate from score-priority NMS.
"""
    peaks = np.asarray(peaks, dtype=float)
    gap = SETTINGS['rulsif']['min_distance'] if min_distance is None else min_distance
    if not isinstance(gap, (int, np.integer)) or gap < 1:
        raise ValueError('minimum alarm distance must be a positive integer')
    positions = np.flatnonzero(np.isfinite(peaks))
    kept = positions[np.r_[True, np.diff(positions) >= gap]] if len(positions) else positions
    result = np.full(len(peaks), np.nan)
    result[kept] = peaks[kept]
    return result


def alarm_predictions(candidates, threshold, bounds):
    start, stop = bounds
    region = np.asarray(candidates)[start:stop]
    return np.flatnonzero(np.isfinite(region) & (region > threshold)) + start


@dataclass
class RuLSIFDetector:
    """Parameterized source estimator; window and threshold selected externally.

Density scores and source plotting positions are retained independently from
the lag-set boundary positions and score availability. No oracle event count.
"""

    n: int = 20
    embedding: int = 10
    alpha: float = .01
    min_distance: int = 1

    def fit(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        required = 2*self.n+self.embedding-1
        if not np.isfinite(values).all() or len(values) < required:
            raise ValueError(f"source RuLSIF requires >={required} finite observations")
        boundary, sigmas, risks, available, plot_positions = score_series(values,self.embedding,n=self.n,alpha=self.alpha)
        indices = np.flatnonzero(np.isfinite(boundary))
        self.boundary_scores_ = boundary
        self.scores_ = boundary.copy()
        self.selected_sigma_ = sigmas
        self.validation_risks_ = risks
        self.score_available_stop_ = available
        self.boundary_positions_ = np.full(len(values), -1, dtype=int)
        self.source_plot_positions_ = np.full(len(values), -1, dtype=int)
        self.boundary_positions_[indices] = indices
        self.source_plot_positions_[indices] = plot_positions
        self.candidate_scores_ = candidate_peak_scores(self.scores_)
        self.alarm_scores_ = chronological_alarm_scores(self.candidate_scores_,self.min_distance)
        return self

    def predict_by_threshold(self, threshold, bounds=None):
        if not hasattr(self,"alarm_scores_"):
            raise ValueError("fit the source detector before predicting")
        return alarm_predictions(self.alarm_scores_, threshold,
                                 bounds or (0,len(self.alarm_scores_)))
