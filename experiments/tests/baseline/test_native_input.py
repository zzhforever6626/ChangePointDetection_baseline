import numpy as np
import pandas as pd
from breakpoint_detection.data import load_yahoo_csv


def test_native_scale_cleaning_keeps_author_noise_units_and_raw_truth(tmp_path):
    path=tmp_path/'A4Benchmark-TS1.csv'
    pd.DataFrame(dict(value=[1000.,1002.,99999.,1006.,1008.],anomaly=[0,0,1,0,0],changepoint=[0,0,1,0,0])).to_csv(path,index=False)
    native=load_yahoo_csv(path,'changepoint','yahoo-anomaly-cleaned')
    normalized=load_yahoo_csv(path,'changepoint','tscp2-a4')
    np.testing.assert_array_equal(native.values,[1000,1002,1004,1006,1008])
    np.testing.assert_array_equal(normalized.values,[0,.25,.5,.75,1])
    np.testing.assert_array_equal(native.labels,[0,0,1,0,0])
    np.testing.assert_array_equal(native.labels,normalized.labels)
