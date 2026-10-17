"""Load all Yahoo A4 files in numeric order for the fixed 40/20/40 split."""
import re
from .data import load_yahoo_csv, yahoo_csv_files


def load_dataset(path, preprocessing="tscp2-a4"):
    files=[p for p in yahoo_csv_files(path) if re.fullmatch(r"A4Benchmark-TS\d+",p.stem,re.I)]
    files.sort(key=lambda p:int(re.search(r"TS(\d+)$",p.stem,re.I).group(1)))
    if not files: raise ValueError("Yahoo A4 files required")
    series=[load_yahoo_csv(p,label_column="changepoint",preprocessing=preprocessing) for p in files]
    if len(files)!=100 or {len(s.values) for s in series}!={1680}:
        raise ValueError("Yahoo A4 requires 100 files with 1680 samples each")
    return files,series
