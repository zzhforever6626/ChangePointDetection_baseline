import numpy as np
import pytest
from breakpoint_detection.alarm_adapter import score_priority_alarm_scores


def test_weak_predecessor_does_not_delete_a_strong_later_peak():
    peaks=np.full(80,np.nan);peaks[[5,15,25,50]]=[1,10,20,2]
    kept=score_priority_alarm_scores(peaks,20)
    assert np.flatnonzero(np.isfinite(kept)).tolist()==[5,25,50]
    peaks[[5,25]]=20
    # Equal scores resolve by earlier position; exactly gap remains eligible.
    assert np.flatnonzero(np.isfinite(score_priority_alarm_scores(peaks,21))).tolist()==[5,50]


@pytest.mark.parametrize('gap',[1,5,20])
def test_suppression_before_threshold_equals_after_threshold(gap):
    rng=np.random.default_rng(12);curve=rng.choice([np.nan,0.,1.,2.],100)
    for threshold in (0.,1.,2.):
        before=score_priority_alarm_scores(curve,gap)
        masked=np.where(curve>threshold,curve,np.nan)
        after=score_priority_alarm_scores(masked,gap)
        np.testing.assert_array_equal(np.flatnonzero(before>threshold),np.flatnonzero(np.isfinite(after)))
