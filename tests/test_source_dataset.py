"""Source-derived partition boundaries and sampled detection availability."""
from pathlib import Path
import numpy as np
from breakpoint_detection.source_dataset import load_dataset
from breakpoint_detection.tscp2_adapter import similarity_drop,phase_peak_scores,training_pairs


def test_pooled_split_preserves_entire_files_and_every_raw_truth():
    root=Path(__file__).resolve().parents[1]
    files,series=load_dataset(root/'data/A4Benchmark')
    assert [p.stem for p in files]==[f'A4Benchmark-TS{i}' for i in range(1,101)]
    for region, expected_truths in [(slice(0,40),86),(slice(40,60),43),(slice(60,100),79)]:
        assert all(len(s.values)==1680 for s in series[region])
        assert sum(np.count_nonzero(s.labels) for s in series[region])==expected_truths
    assert sum(np.count_nonzero(s.labels) for s in series)==208


def test_source_exclusive_windows_and_sampled_moving_mean():
    pairs,locations=training_pairs(np.arange(40)[None,:],[40],5,5,last_start_inclusive=False)
    assert locations[:,1].tolist()==[0,5,10,15,20,25]
    np.testing.assert_array_equal(pairs[-1],np.arange(25,35))
    raw=np.full(60,np.nan); raw[5::5]=[1,1,1,0,1,1,1,1,1,1,1]
    scores=similarity_drop(raw,3,step=5)
    np.testing.assert_allclose(scores[[20,25]],[1,-1/3])
    peaks=phase_peak_scores(scores,3,35,step=5)
    # The first finite score lacks its left score, so is not a peak.
    assert np.isnan(peaks[20])
    assert np.flatnonzero(np.isfinite(peaks)).max(initial=0)+3+5<=35


def test_training_pairs_do_not_use_validation_or_test_values():
    values=np.arange(100*40).reshape(100,40)
    train_stops=[40]*40+[0]*60
    expected=training_pairs(values,train_stops,5,last_start_inclusive=False)
    changed=values.copy();changed[40:]=-999
    actual=training_pairs(changed,train_stops,5,last_start_inclusive=False)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_array_equal(actual[1],expected[1])
