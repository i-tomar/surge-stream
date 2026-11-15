from __future__ import annotations

from datetime import datetime

import numpy as np
from surgecast.config import NY, ORIGIN_EPOCH
from surgecast.data.prepare import counts_from_trips
from surgecast.features.compute import (
    FEATURE_COLUMNS,
    bucket_of,
    epoch_of_bucket,
    feature_rows,
    slice_sum,
)
from surgecast.streaming.processor import FeatureProcessor


def test_known_window_does_not_leak_future_counts():
    counts = np.zeros(3000, dtype=np.int32)
    t = 2500
    counts[2499] = 2
    counts[2498] = 3
    counts[2497] = 4
    counts[2480] = 9
    counts[2212] = 11
    counts[2213] = 1
    counts[2214] = 1
    counts[2500] = 8
    counts[2501] = 8
    counts[2502] = 8
    row = feature_rows(counts, np.array([t]), 42, ORIGIN_EPOCH)
    assert row["zone_id"][0] == 42
    assert row["pu_5m"][0] == 2
    assert row["pu_15m"][0] == 9
    assert row["pu_30m"][0] == 9
    assert row["pu_60m"][0] == 9
    assert row["pu_lag15"][0] == 0
    assert row["pu_lag60"][0] == 9
    assert row["seasonal_yday_15"][0] == 13
    assert row["delta_15"][0] == 9
    assert row["trend_ratio"][0] == 4.0
    assert row["y15"][0] == 24
    assert set(FEATURE_COLUMNS).issubset(row)


def test_clock_features_use_new_york_time():
    t = 2500
    row = feature_rows(np.zeros(3000, dtype=np.int32), np.array([t]), 1, ORIGIN_EPOCH)
    local = datetime.fromtimestamp(ORIGIN_EPOCH + t * 300, tz=NY)
    assert row["hour"][0] == local.hour
    assert row["dow"][0] == local.weekday()
    assert row["slot_15"][0] == (local.hour * 60 + local.minute) // 15
    assert row["is_weekend"][0] == float(local.weekday() >= 5)


def test_vector_matches_scalar_slice_sum():
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 6, size=4000, dtype=np.int32)
    indexes = rng.integers(2016, 3990, size=25)
    rows = feature_rows(counts, indexes, 7, ORIGIN_EPOCH)
    for i, t in enumerate(indexes):
        assert rows["pu_15m"][i] == slice_sum(counts, t - 3, t)
        assert rows["pu_60m"][i] == slice_sum(counts, t - 12, t)
        assert rows["pu_lag15"][i] == slice_sum(counts, t - 6, t - 3)
        assert rows["seasonal_yday_15"][i] == slice_sum(counts, t - 288, t - 285)
        assert rows["seasonal_week_15"][i] == slice_sum(counts, t - 2016, t - 2013)
        assert rows["y15"][i] == slice_sum(counts, t, t + 3)
        assert rows["y30"][i] == slice_sum(counts, t, t + 6)


def test_batch_counts_match_online_store():
    import fakeredis

    origin = ORIGIN_EPOCH
    epochs = np.array(
        [epoch_of_bucket(10, origin) + 5, epoch_of_bucket(10, origin) + 8, epoch_of_bucket(12, origin) + 1],
        dtype=np.int64,
    )
    zones = np.array([4, 4, 4], dtype=np.int32)
    batch = counts_from_trips(epochs, zones, origin)
    client = fakeredis.FakeRedis(decode_responses=True)
    processor = FeatureProcessor(client, origin_epoch=origin)
    previous = processor.max_bucket
    advanced = False
    for ts, zone in zip(epochs, zones, strict=True):
        if processor.handle_event(int(ts), int(zone)) is not None:
            advanced = True
    assert advanced
    processor.flush_range(previous, received_at=0.0)
    assert processor.store.counts[4, 10] == batch[4, 10] == 2
    assert processor.store.counts[4, 12] == batch[4, 12] == 1
    raw = client.hgetall("zone:4")
    # Decision index is the open bucket (12). pu_5m is the previous bucket.
    assert int(raw["bucket"]) == 12
    assert float(raw["pu_5m"]) == 0
    assert float(raw["pu_15m"]) == 2
    direct = feature_rows(processor.store.counts[4], np.array([12]), 4, origin)
    assert float(raw["pu_15m"]) == direct["pu_15m"][0]
    assert float(raw["pu_60m"]) == direct["pu_60m"][0]
    assert bucket_of(int(epochs[0]), origin) == 10
