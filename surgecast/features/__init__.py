"""Feature definitions shared by offline training and the online processor."""

from surgecast.features.compute import (
    DRIFT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    feature_rows,
    slice_sum,
)

__all__ = [
    "DRIFT_COLUMNS",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "feature_rows",
    "slice_sum",
]
