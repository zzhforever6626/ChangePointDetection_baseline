"""Author window-pool split structure with the 2024 40/20/40 proportions."""
import re
import numpy as np
from .data import load_yahoo_csv, yahoo_csv_files
from .experiment import TemporalSplit, yahoo_a4_split


def load_dataset(path, layout="source-pooled",preprocessing="tscp2-a4"):
    files=[p for p in yahoo_csv_files(path) if re.fullmatch(r"A4Benchmark-TS\d+",p.stem,re.I)]
    files.sort(key=lambda p:int(re.search(r"TS(\d+)$",p.stem,re.I).group(1)))
    if not files: raise ValueError("Yahoo A4 files required")
    series=[load_yahoo_csv(p,label_column="changepoint",preprocessing=preprocessing) for p in files]
    if layout=="per-series":
        splits=[yahoo_a4_split(len(s.values)) for s in series]
    elif layout=="source-pooled":
        # Equal Yahoo series contribute equal counts for each fixed window,
        # making the 40/20/40 window-pool boundaries whole-series boundaries.
        if len(files)!=100 or {len(s.values) for s in series}!={1680}:
            raise ValueError("registered pooled Yahoo layout requires 100 equally sized A4 series; use --dataset-layout per-series for research fixtures")
        splits=[]
        for index,s in enumerate(series):
            bounds=[(0,0)]*3
            bounds[0 if index<40 else 1 if index<60 else 2]=(0,len(s.values))
            splits.append(TemporalSplit(*bounds))
    else: raise ValueError("unknown dataset layout")
    return files,series,splits


def phase_inputs(series,splits,phase):
    indices=[i for i,split in enumerate(splits) if getattr(split,phase)[1]>getattr(split,phase)[0]]
    bounds=[getattr(splits[i],phase) for i in indices]
    truth=[np.flatnonzero(series[i].labels[start:stop])+start for i,(start,stop) in zip(indices,bounds)]
    return indices,truth,bounds
