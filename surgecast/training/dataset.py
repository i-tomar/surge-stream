"""Build the supervised frame from bucket counts using ``feature_rows``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surgecast.config import ORIGIN_EPOCH, TEST_START_DT, TRAIN_MIN_BUCKET, VAL_START_DT
from surgecast.features.compute import FEATURE_COLUMNS, LABEL_COLUMNS, feature_rows

FRAME_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS + ["bucket", "decision_epoch"]


def build_frame(
    counts: np.ndarray,
    origin_epoch: int = ORIGIN_EPOCH,
    min_bucket: int = TRAIN_MIN_BUCKET,
    max_bucket: int | None = None,
) -> pd.DataFrame:
    """One row per (zone, decision bucket) for zones that have any pickups.

    ``max_bucket`` is exclusive. It defaults to the last index whose 30-minute
    label lies inside the count array.
    """
    n_buckets = int(counts.shape[1])
    if max_bucket is None:
        max_bucket = n_buckets - 6
    max_bucket = min(int(max_bucket), n_buckets - 6)
    if max_bucket <= min_bucket:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    t_index = np.arange(min_bucket, max_bucket, dtype=np.int64)
    parts: list[pd.DataFrame] = []
    for zone in range(1, counts.shape[0]):
        if int(counts[zone].sum()) == 0:
            continue
        columns = feature_rows(counts[zone], t_index, zone, origin_epoch)
        parts.append(pd.DataFrame(columns))
    if not parts:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    frame = pd.concat(parts, ignore_index=True)
    frame["zone_id"] = frame["zone_id"].astype(np.int32)
    return frame


def time_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test cut on the decision timestamp. No shuffle."""
    val_epoch = int(VAL_START_DT.timestamp())
    test_epoch = int(TEST_START_DT.timestamp())
    epoch = frame["decision_epoch"].to_numpy()
    train = frame.loc[epoch < val_epoch]
    val = frame.loc[(epoch >= val_epoch) & (epoch < test_epoch)]
    test = frame.loc[epoch >= test_epoch]
    return train, val, test


def matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    features["zone_id"] = features["zone_id"].astype(np.int32)
    y15 = frame["y15"].to_numpy(dtype=np.float64)
    y30 = frame["y30"].to_numpy(dtype=np.float64)
    return features, y15, y30
