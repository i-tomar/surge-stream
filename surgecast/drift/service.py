"""Poll Redis feature samples, run Evidently, and retrain on drift."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import orjson
import pandas as pd
import redis

from surgecast.config import (
    DRIFT_CHECK_SEC,
    DRIFT_COOLDOWN_SEC,
    DRIFT_MIN_ROWS,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
    REDIS_URL,
    RUNTIME_DIR,
    ensure_dirs,
)
from surgecast.drift.detect import detect_drift, should_retrain
from surgecast.drift.retrain import retrain_from_counts
from surgecast.features.compute import FEATURE_COLUMNS
from surgecast.metrics.prom import (
    DRIFT_DETECTED,
    DRIFT_SCORE,
    MODEL_VERSION,
    PREDICTION_ABS_ERROR,
    serve_metrics,
)
from surgecast.serving.model_loader import champion_versions, load_champions
from surgecast.training.metrics import clip_prediction

logger = logging.getLogger("surgecast.drift")


def _frame_from_list(client: redis.Redis, key: str, limit: int = 1000) -> pd.DataFrame:
    raw = client.lrange(key, 0, limit - 1)
    if not raw:
        return pd.DataFrame()
    rows = []
    for item in raw:
        rows.append(orjson.loads(item))
    return pd.DataFrame(rows)


class _ModelCache:
    def __init__(self) -> None:
        self.loaded = None

    def get(self):
        try:
            versions = champion_versions()
        except Exception as exc:
            logger.warning("champion lookup failed: %s", exc)
            return self.loaded
        if self.loaded is not None and (self.loaded[0], self.loaded[2]) == versions:
            return self.loaded
        self.loaded = load_champions()
        return self.loaded


def _score_labeled(frame: pd.DataFrame, champions) -> None:
    if frame.empty or champions is None or "y15" not in frame.columns:
        return
    version_15, model_15, version_30, model_30 = champions
    usable = frame.dropna(subset=["y15"])
    if model_15 is not None and len(usable) >= 50:
        features = usable.loc[:, FEATURE_COLUMNS].copy()
        features["zone_id"] = features["zone_id"].astype(np.int32)
        pred = clip_prediction(model_15.predict(features))
        err = float(np.mean(np.abs(pred - usable["y15"].to_numpy(dtype=float))))
        PREDICTION_ABS_ERROR.labels(horizon="15").set(err)
        if version_15 is not None:
            MODEL_VERSION.labels(horizon="15").set(version_15)
    if "y30" not in frame.columns or model_30 is None:
        return
    usable_30 = frame.dropna(subset=["y30"])
    if len(usable_30) >= 50:
        features = usable_30.loc[:, FEATURE_COLUMNS].copy()
        features["zone_id"] = features["zone_id"].astype(np.int32)
        pred = clip_prediction(model_30.predict(features))
        err = float(np.mean(np.abs(pred - usable_30["y30"].to_numpy(dtype=float))))
        PREDICTION_ABS_ERROR.labels(horizon="30").set(err)
        if version_30 is not None:
            MODEL_VERSION.labels(horizon="30").set(version_30)


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


def _write_status(status: dict) -> None:
    (RUNTIME_DIR / "drift_status.json").write_text(json.dumps(status, indent=2, default=_default))


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    serve_metrics(int(os.environ.get("METRICS_PORT", "8844")))
    client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    reference_path = PROCESSED_DIR / "reference_features.parquet"
    models = _ModelCache()
    last_retrain: float | None = None
    while True:
        try:
            status: dict = {"checked_at": datetime.now(timezone.utc).isoformat()}
            if not reference_path.exists():
                status["state"] = "waiting_for_reference"
                _write_status(status)
                time.sleep(DRIFT_CHECK_SEC)
                continue
            current = _frame_from_list(client, "features:recent")
            labeled = _frame_from_list(client, "labeled:recent")
            try:
                _score_labeled(labeled, models.get())
            except Exception as exc:
                status["score_error"] = str(exc)
            if len(current) < DRIFT_MIN_ROWS:
                status.update({"state": "warming", "rows": int(len(current)), "need": DRIFT_MIN_ROWS})
                DRIFT_DETECTED.set(0)
                _write_status(status)
                time.sleep(DRIFT_CHECK_SEC)
                continue
            reference = pd.read_parquet(reference_path)
            report = detect_drift(reference, current)
            DRIFT_SCORE.set(report.share_drifted)
            now = time.time()
            retrain_now = should_retrain(report, last_retrain, now, DRIFT_COOLDOWN_SEC)
            DRIFT_DETECTED.set(1 if retrain_now else 0)
            status.update(
                {
                    "state": "drift" if report.dataset_drift else "stable",
                    "share_drifted": report.share_drifted,
                    "detect_seconds": report.seconds,
                    "rows": int(len(current)),
                    "retrain": retrain_now,
                }
            )
            if retrain_now:
                detected_unix = time.time()
                counts_path = RUNTIME_DIR / "live_counts.npz"
                if not counts_path.exists():
                    status["retrain_error"] = "live_counts.npz missing"
                else:
                    counts = np.load(counts_path)["counts"]
                    result = retrain_from_counts(counts, reason=f"evidently share={report.share_drifted:.3f}")
                    promoted_unix = time.time()
                    last_retrain = promoted_unix
                    models.loaded = None
                    recovery = {
                        "detected_unix": detected_unix,
                        "promoted_unix": promoted_unix,
                        "detect_to_promote_seconds": promoted_unix - detected_unix,
                        **result,
                    }
                    (RUNTIME_DIR / "recovery.json").write_text(json.dumps(recovery, indent=2, default=_default))
                    status["recovery"] = {
                        "detect_to_promote_seconds": recovery["detect_to_promote_seconds"],
                        "version_15": result["version_15"],
                        "version_30": result["version_30"],
                    }
                    logger.info("promoted 15m version %s", result["version_15"])
            _write_status(status)
        except Exception:
            logger.exception("drift loop failed")
        time.sleep(DRIFT_CHECK_SEC)


if __name__ == "__main__":
    run()
