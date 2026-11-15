import pandas as pd
from surgecast.config import NY, ORIGIN_EPOCH


def test_microsecond_timestamps_convert_to_unix_seconds():
    """TLC parquet timestamps arrive as datetime64[us]. Dividing by 1e9 would land in 1970."""
    raw = pd.Series(pd.to_datetime(["2024-01-01 00:00:00"])).astype("datetime64[us]")
    local = raw.dt.tz_localize(NY)
    divisor = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}[str(local.dtype.unit)]
    epoch = int(local.astype("int64").iloc[0] // divisor)
    assert epoch == ORIGIN_EPOCH
