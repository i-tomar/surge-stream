"""Runtime configuration shared by training, streaming, and serving."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# Midnight at the start of the demo month. Bucket 0 opens at this instant.
# January is entirely EST (UTC-5), so the origin does not cross a DST change.
ORIGIN_DT = datetime(2024, 1, 1, tzinfo=NY)
ORIGIN_EPOCH = int(ORIGIN_DT.timestamp())

BUCKET_SECONDS = 300
DAY_BUCKETS = 24 * 60 // 5  # 288
WEEK_BUCKETS = 7 * DAY_BUCKETS  # 2016
N_ZONES = 263
SCHEMA_VERSION = 1

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SURGECAST_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNTIME_DIR = DATA_DIR / "runtime"
RESULTS_DIR = ROOT / "results"

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "trips")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5055")

MODEL_NAME_15 = "surgecast_demand_15m"
MODEL_NAME_30 = "surgecast_demand_30m"
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "surgecast-demand"

# Rows with a full week of history are the first eligible training rows.
# That is 2024-01-08 00:00 America/New_York.
TRAIN_MIN_BUCKET = WEEK_BUCKETS
VAL_START_DT = datetime(2024, 1, 26, tzinfo=NY)
TEST_START_DT = datetime(2024, 1, 28, tzinfo=NY)

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

DRIFT_SHARE_THRESHOLD = 0.5
DRIFT_PSI_THRESHOLD = 0.2
DRIFT_MIN_ROWS = 400
DRIFT_COOLDOWN_SEC = 600
DRIFT_CHECK_SEC = 10


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, RUNTIME_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
