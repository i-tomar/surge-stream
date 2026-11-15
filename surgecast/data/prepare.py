"""Turn TLC yellow-taxi pickups into per-zone 5-minute counts.

Timestamps in the parquet file are naive America/New_York wall times. Rows
outside January 2024, or with a pickup zone outside 1..263, are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from surgecast.config import (
    BUCKET_SECONDS,
    N_ZONES,
    NY,
    ORIGIN_EPOCH,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
)

MONTH_START = pd.Timestamp("2024-01-01")
MONTH_END = pd.Timestamp("2024-02-01")


def trips_path() -> Path:
    return RAW_DIR / "yellow_tripdata_2024-01.parquet"


def counts_path() -> Path:
    return PROCESSED_DIR / "bucket_counts.npz"


def meta_path() -> Path:
    return PROCESSED_DIR / "bucket_counts_meta.json"


def load_trips(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(epoch_seconds, zone_id, stats)`` for kept trips, sorted by time."""
    path = path or trips_path()
    frame = pd.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
    n_raw = int(len(frame))
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"])
    zone = pd.to_numeric(frame["PULocationID"], errors="coerce")
    keep = pickup.notna() & zone.notna()
    keep &= (pickup >= MONTH_START) & (pickup < MONTH_END)
    keep &= (zone >= 1) & (zone <= N_ZONES)
    pickup = pickup[keep]
    zone = zone[keep].astype(np.int32)
    local = pickup.dt.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward")
    valid = local.notna()
    local = local[valid]
    unit = str(getattr(local.dtype, "unit", "ns"))
    divisor = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}[unit]
    epoch = (local.astype("int64") // divisor).to_numpy(dtype=np.int64)
    zone_arr = zone.to_numpy(dtype=np.int32)[valid.to_numpy()]
    order = np.argsort(epoch, kind="mergesort")
    epoch = epoch[order]
    zone_arr = zone_arr[order]
    stats = {
        "n_raw": n_raw,
        "n_kept": int(epoch.shape[0]),
        "min_epoch": int(epoch[0]) if len(epoch) else None,
        "max_epoch": int(epoch[-1]) if len(epoch) else None,
        "source": path.name,
    }
    return epoch, zone_arr, stats


def counts_from_trips(epochs: np.ndarray, zones: np.ndarray, origin_epoch: int = ORIGIN_EPOCH) -> np.ndarray:
    """Aggregate trips with the same bucket index the stream processor uses."""
    if len(epochs) == 0:
        return np.zeros((N_ZONES + 1, 1), dtype=np.int32)
    buckets = (epochs.astype(np.int64) - int(origin_epoch)) // BUCKET_SECONDS
    valid = buckets >= 0
    buckets = buckets[valid]
    zones = np.asarray(zones)[valid]
    if len(buckets) == 0:
        return np.zeros((N_ZONES + 1, 1), dtype=np.int32)
    n_buckets = int(buckets.max()) + 1
    counts = np.zeros((N_ZONES + 1, n_buckets), dtype=np.int32)
    np.add.at(counts, (zones.astype(np.int64), buckets), 1)
    return counts


def build_counts(path: Path | None = None) -> tuple[np.ndarray, dict]:
    ensure_dirs()
    epochs, zones, stats = load_trips(path)
    counts = counts_from_trips(epochs, zones, ORIGIN_EPOCH)
    meta = {
        **stats,
        "origin_epoch": ORIGIN_EPOCH,
        "n_buckets": int(counts.shape[1]),
        "n_zones": N_ZONES,
        "bucket_seconds": 300,
    }
    np.savez_compressed(counts_path(), counts=counts)
    meta_path().write_text(json.dumps(meta, indent=2))
    return counts, meta


def load_counts() -> tuple[np.ndarray, dict]:
    if not counts_path().exists():
        return build_counts()
    counts = np.load(counts_path())["counts"]
    meta = json.loads(meta_path().read_text())
    return counts, meta
