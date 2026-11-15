"""Rolling taxi-zone features.

Offline training and the online processor both call ``feature_rows``. The
function only reads buckets strictly before the decision index ``t``, and the
labels are the buckets at and after ``t``. Keeping a single implementation is
what prevents train/serve skew.

Bucket ``b`` covers ``[origin + b * 300s, origin + (b + 1) * 300s)``.
A decision at index ``t`` is made at the start of bucket ``t`` and forecasts
``counts[t : t + horizon]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surgecast.config import BUCKET_SECONDS, DAY_BUCKETS, NY, WEEK_BUCKETS

FEATURE_COLUMNS: list[str] = [
    "zone_id",
    "pu_5m",
    "pu_15m",
    "pu_30m",
    "pu_60m",
    "pu_lag15",
    "pu_lag60",
    "seasonal_yday_15",
    "seasonal_week_15",
    "seasonal_yday_30",
    "seasonal_week_30",
    "delta_15",
    "trend_ratio",
    "hour",
    "dow",
    "is_weekend",
    "slot_15",
]

# Columns whose distribution the drift monitor compares.
DRIFT_COLUMNS: list[str] = [
    "pu_5m",
    "pu_15m",
    "pu_30m",
    "pu_60m",
    "pu_lag15",
    "pu_lag60",
    "seasonal_yday_15",
    "delta_15",
    "trend_ratio",
]

LABEL_COLUMNS = ["y15", "y30"]


def slice_sum(counts: np.ndarray, start: int, end: int) -> int:
    """Sum ``counts[start:end]``, clipping to the array and returning 0 when empty."""
    n = int(counts.shape[0])
    if end <= 0 or start >= n:
        return 0
    start_i = max(int(start), 0)
    end_i = min(int(end), n)
    if start_i >= end_i:
        return 0
    return int(counts[start_i:end_i].sum())


def _window(cs: np.ndarray, n: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start_a, end_a = np.broadcast_arrays(np.asarray(start, dtype=np.int64), np.asarray(end, dtype=np.int64))
    s = np.clip(start_a, 0, n)
    e = np.clip(end_a, 0, n)
    out = (cs[e] - cs[s]).astype(np.float64, copy=True)
    out[s >= e] = 0.0
    return out


def feature_rows(
    counts: np.ndarray,
    t_index: np.ndarray,
    zone_id: int,
    origin_epoch: int,
) -> dict[str, np.ndarray]:
    """Features and labels for one zone at each decision index in ``t_index``.

    ``counts[b]`` is the pickup count in bucket ``b``. Labels use future buckets
    and are only defined once those buckets have closed; callers that need a
    feature vector at decision time simply ignore ``y15`` and ``y30``.
    """
    t = np.asarray(t_index, dtype=np.int64)
    c = np.asarray(counts, dtype=np.int64)
    n = int(c.shape[0])
    cs = np.empty(n + 1, dtype=np.int64)
    cs[0] = 0
    if n:
        np.cumsum(c, out=cs[1:])

    def w(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return _window(cs, n, start, end)

    pu_5 = w(t - 1, t)
    pu_15 = w(t - 3, t)
    pu_30 = w(t - 6, t)
    pu_60 = w(t - 12, t)
    pu_lag15 = w(t - 6, t - 3)
    pu_lag60 = w(t - 24, t - 12)
    seasonal_yday_15 = w(t - DAY_BUCKETS, t - DAY_BUCKETS + 3)
    seasonal_week_15 = w(t - WEEK_BUCKETS, t - WEEK_BUCKETS + 3)
    seasonal_yday_30 = w(t - DAY_BUCKETS, t - DAY_BUCKETS + 6)
    seasonal_week_30 = w(t - WEEK_BUCKETS, t - WEEK_BUCKETS + 6)
    delta_15 = pu_15 - pu_lag15
    hourly = pu_60 / 4.0
    trend = np.divide(pu_15, hourly, out=np.zeros_like(pu_15), where=hourly > 0.0)

    decision = origin_epoch + t * BUCKET_SECONDS
    if len(t) == 0:
        hour = np.zeros(0, dtype=np.float64)
        dow = np.zeros(0, dtype=np.float64)
        slot = np.zeros(0, dtype=np.float64)
        weekend = np.zeros(0, dtype=np.float64)
    else:
        local = pd.to_datetime(decision, unit="s", utc=True).tz_convert(NY)
        hour = local.hour.to_numpy(dtype=np.float64)
        dow = local.dayofweek.to_numpy(dtype=np.float64)
        slot = ((local.hour.to_numpy() * 60 + local.minute.to_numpy()) // 15).astype(np.float64)
        weekend = (dow >= 5).astype(np.float64)

    return {
        "zone_id": np.full(t.shape, int(zone_id), dtype=np.int32),
        "pu_5m": pu_5,
        "pu_15m": pu_15,
        "pu_30m": pu_30,
        "pu_60m": pu_60,
        "pu_lag15": pu_lag15,
        "pu_lag60": pu_lag60,
        "seasonal_yday_15": seasonal_yday_15,
        "seasonal_week_15": seasonal_week_15,
        "seasonal_yday_30": seasonal_yday_30,
        "seasonal_week_30": seasonal_week_30,
        "delta_15": delta_15,
        "trend_ratio": trend,
        "hour": hour,
        "dow": dow,
        "is_weekend": weekend,
        "slot_15": slot,
        "y15": w(t, t + 3),
        "y30": w(t, t + 6),
        "bucket": t.astype(np.int64),
        "decision_epoch": decision.astype(np.int64),
    }


def bucket_of(epoch: int, origin_epoch: int) -> int:
    return (int(epoch) - int(origin_epoch)) // BUCKET_SECONDS


def epoch_of_bucket(bucket: int, origin_epoch: int) -> int:
    return int(origin_epoch) + int(bucket) * BUCKET_SECONDS
