"""Validation-fitted event selection; not a published density-ratio parameter."""
import numpy as np


def score_priority_alarm_scores(peaks,gap):
    """Keep strongest peaks, earliest on exact ties; distance >= gap survives.

    Suppression precedes thresholding. Because only stronger retained peaks
    suppress weaker peaks, this equals suppressing after a strict threshold.
    No ground truth or desired event count is used.
    """
    if gap<1:raise ValueError("gap must be positive")
    peaks=np.asarray(peaks,dtype=float)
    positions=np.flatnonzero(np.isfinite(peaks))
    order=sorted(positions,key=lambda position:(-peaks[position],position))
    allowed=np.ones(len(peaks),dtype=bool);kept=[]
    for position in order:
        if allowed[position]:
            kept.append(position)
            allowed[max(0,position-gap+1):min(len(peaks),position+gap)]=False
    result=np.full(peaks.shape,np.nan);result[kept]=peaks[kept]
    return result
