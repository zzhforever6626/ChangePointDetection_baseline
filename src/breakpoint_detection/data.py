from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeries:
    values: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    name: str
    label_column: str
    preprocessing: str


def load_yahoo_csv(
    path: str | Path, label_column: str = "auto", preprocessing: str = "none"
) -> TimeSeries:
    """Load a Yahoo S5 CSV and return a common ``(X, y)`` representation.

    Yahoo A4 files contain both ``anomaly`` (outliers) and ``changepoint``.
    ``auto`` prefers the latter; use an explicit label column for a benchmark.
    A1/A2 files contain ``is_anomaly`` instead.
    Missing values are linearly interpolated, including at either edge.
    ``tscp2-a4`` additionally replaces A4 spatial anomalies with neighboring
    values and min-max normalizes each series, as in the TS-CP2 data helper.
    ``yahoo-anomaly-cleaned`` performs the same replacement while retaining
    native units, so the FLOSS author's .001 noise is applied before any plot scaling.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    lower = {str(column).strip().lower(): column for column in frame.columns}

    def find(*names: str, required: bool = True):
        for name in names:
            if name in lower:
                return lower[name]
        if required:
            raise ValueError(f"{path}: missing one of columns {names}; got {list(frame.columns)}")
        return None

    value_col = find("value", "values")
    if label_column == "auto":
        label_col = find("changepoint", "is_changepoint", "change_point",
                         "is_anomaly", "anomaly", "label")
    else:
        label_col = find(label_column.strip().lower())
    time_col = find("timestamp", "timestamps", "time", required=False)

    values = pd.to_numeric(frame[value_col], errors="coerce")
    values = values.interpolate(limit_direction="both")
    if values.isna().any():
        raise ValueError(f"{path}: value column contains no usable numeric data")
    values_array = values.to_numpy(dtype=np.float64, copy=True)
    if preprocessing in ("tscp2-a4","yahoo-anomaly-cleaned"):
        anomaly_col = find("anomaly")
        anomaly = pd.to_numeric(frame[anomaly_col], errors="coerce")
        if anomaly.isna().any() or not anomaly.isin((0, 1)).all():
            raise ValueError(f"{path}: anomaly column must contain only 0/1 values")
        if len(values_array) < 2 or anomaly.all():
            raise ValueError(f"{path}: cannot replace anomalies in this series")
        for index in np.flatnonzero(anomaly.to_numpy(dtype=bool)):
            if index == 0:
                values_array[index] = values_array[index + 1]
            elif index == len(values_array) - 1:
                values_array[index] = values_array[index - 1]
            else:
                values_array[index] = 0.5 * (values_array[index - 1] + values_array[index + 1])
        if preprocessing == "tscp2-a4":
            minimum, maximum = float(values_array.min()), float(values_array.max())
            values_array = ((values_array - minimum) / (maximum - minimum)
                            if maximum > minimum else np.zeros_like(values_array))
    elif preprocessing != "none":
        raise ValueError("preprocessing must be none, tscp2-a4 or yahoo-anomaly-cleaned")

    labels = pd.to_numeric(frame[label_col], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError(f"{path}: label column {label_col!r} must contain only 0/1 values")
    timestamps = frame[time_col].to_numpy() if time_col else np.arange(len(frame))
    if len(values) == 0:
        raise ValueError(f"{path}: empty time series")
    return TimeSeries(
        values=values_array,
        labels=labels.to_numpy(dtype=np.int8),
        timestamps=np.asarray(timestamps),
        name=path.stem,
        label_column=str(label_col),
        preprocessing=preprocessing,
    )


def yahoo_csv_files(path: str | Path) -> list[Path]:
    """Return one CSV or all CSV files below a directory, deterministically."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found below {path}")
    return files
