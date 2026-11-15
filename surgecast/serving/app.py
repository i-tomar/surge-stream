"""FastAPI predictions from Redis features and the MLflow champion models."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from surgecast.config import N_ZONES, NY, RAW_DIR, REDIS_URL
from surgecast.features.compute import FEATURE_COLUMNS
from surgecast.metrics.prom import API_LATENCY, MODEL_VERSION, serve_metrics
from surgecast.serving.model_loader import champion_versions, load_champions
from surgecast.training.metrics import clip_prediction

logger = logging.getLogger("surgecast.api")


class ChampionCache:
    def __init__(self, loader=load_champions, versions=champion_versions) -> None:
        self.loader = loader
        self.versions = versions
        self.version_15 = None
        self.version_30 = None
        self.model_15 = None
        self.model_30 = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def refresh(self) -> bool:
        try:
            v15, v30 = self.versions()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        with self._lock:
            unchanged = (
                v15 == self.version_15
                and v30 == self.version_30
                and self.model_15 is not None
                and self.model_30 is not None
            )
        if unchanged:
            self.last_error = None
            return False
        try:
            loaded = self.loader()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        with self._lock:
            self.version_15, self.model_15, self.version_30, self.model_30 = loaded
            self.last_error = None
        MODEL_VERSION.labels(horizon="15").set(loaded[0])
        MODEL_VERSION.labels(horizon="30").set(loaded[2])
        return True

    def snapshot(self):
        with self._lock:
            return self.version_15, self.model_15, self.version_30, self.model_30


def _zone_names(path) -> dict[int, tuple[str, str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    names = {}
    for row in frame.itertuples(index=False):
        names[int(row.LocationID)] = (str(row.Zone), str(row.Borough))
    return names


def _frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["zone_id"] = frame["zone_id"].astype(np.int32)
    for column in FEATURE_COLUMNS:
        if column != "zone_id":
            frame[column] = frame[column].astype(np.float64)
    return frame.loc[:, FEATURE_COLUMNS]


def _record_from_hash(zone_id: int, raw: dict) -> dict:
    missing = [column for column in FEATURE_COLUMNS if column != "zone_id" and column not in raw]
    if missing:
        raise KeyError(",".join(missing))
    record = {"zone_id": zone_id}
    for column in FEATURE_COLUMNS:
        if column == "zone_id":
            continue
        record[column] = float(raw[column])
    return record


def create_app(
    redis_client: redis.Redis | None = None,
    cache: ChampionCache | None = None,
    start_background: bool = True,
    zone_table: dict | None = None,
) -> FastAPI:
    app = FastAPI(title="Surgecast", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    client = redis_client if redis_client is not None else redis.Redis.from_url(REDIS_URL, decode_responses=True)
    champions = cache or ChampionCache()
    names = zone_table if zone_table is not None else _zone_names(RAW_DIR / "taxi_zone_lookup.csv")
    stop = threading.Event()

    @app.middleware("http")
    async def time_request(request, call_next):
        started = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", request.url.path)
            API_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)

    def _refresh_loop() -> None:
        while not stop.wait(2.0):
            try:
                if champions.refresh():
                    logger.info("loaded champions %s %s", champions.version_15, champions.version_30)
            except Exception:
                logger.exception("champion refresh failed")

    @app.on_event("startup")
    def startup() -> None:
        if start_background:
            threading.Thread(target=_refresh_loop, name="champion-refresh", daemon=True).start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        stop.set()

    def _meta() -> dict:
        try:
            watermark = client.get("meta:watermark")
            bucket = client.get("meta:bucket")
            return {
                "redis": True,
                "watermark_epoch": int(watermark) if watermark else None,
                "bucket": int(bucket) if bucket else None,
            }
        except Exception as exc:
            return {"redis": False, "error": str(exc), "watermark_epoch": None, "bucket": None}

    @app.get("/api/health")
    def health():
        v15, _, v30, _ = champions.snapshot()
        meta = _meta()
        watermark = meta.get("watermark_epoch")
        event_time = None
        if watermark:
            event_time = datetime.fromtimestamp(watermark, tz=NY).isoformat()
        ready = v15 is not None and v30 is not None and meta.get("redis")
        return {
            "status": "ok" if ready else "degraded",
            "model_15": v15,
            "model_30": v30,
            "model_error": champions.last_error,
            "event_time": event_time,
            **meta,
        }

    @app.post("/api/admin/reload")
    def reload_models():
        changed = champions.refresh()
        v15, _, v30, _ = champions.snapshot()
        if v15 is None or v30 is None:
            raise HTTPException(status_code=503, detail=champions.last_error or "model not loaded")
        return {"reloaded": changed, "model_15": v15, "model_30": v30}

    def _predict_frame(frame: pd.DataFrame, horizon: int):
        v15, m15, v30, m30 = champions.snapshot()
        model = m15 if horizon == 15 else m30
        version = v15 if horizon == 15 else v30
        if model is None:
            raise HTTPException(status_code=503, detail=champions.last_error or "model not loaded")
        prediction = clip_prediction(model.predict(frame))
        return version, prediction

    @app.get("/api/predict/{zone_id}")
    def predict(zone_id: int, horizon: int = Query(default=15)):
        if horizon not in (15, 30):
            raise HTTPException(status_code=422, detail="horizon must be 15 or 30")
        if zone_id < 1 or zone_id > N_ZONES:
            raise HTTPException(status_code=404, detail="unknown zone")
        raw = client.hgetall(f"zone:{zone_id}")
        if not raw:
            raise HTTPException(status_code=404, detail="zone has no features yet")
        try:
            record = _record_from_hash(zone_id, raw)
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=f"incomplete feature hash: {exc}") from exc
        frame = _frame([record])
        version, prediction = _predict_frame(frame, horizon)
        name, borough = names.get(zone_id, (f"Zone {zone_id}", ""))
        return {
            "zone_id": zone_id,
            "name": name,
            "borough": borough,
            "horizon_min": horizon,
            "prediction": float(prediction[0]),
            "model_version": version,
            "pu_15m": record["pu_15m"],
            "pu_60m": record["pu_60m"],
            "decision_epoch": int(float(raw.get("decision_epoch", 0))),
        }

    @app.get("/api/zones")
    def zones():
        pipe = client.pipeline(transaction=False)
        for zone_id in range(1, N_ZONES + 1):
            pipe.hgetall(f"zone:{zone_id}")
        raws = pipe.execute()
        records = []
        kept = []
        for zone_id, raw in enumerate(raws, start=1):
            if not raw:
                continue
            try:
                record = _record_from_hash(zone_id, raw)
            except KeyError:
                continue
            records.append(record)
            kept.append((zone_id, raw, record))
        pred15 = pred30 = None
        v15 = v30 = None
        if records:
            frame = _frame(records)
            try:
                v15, pred15 = _predict_frame(frame, 15)
                v30, pred30 = _predict_frame(frame, 30)
            except HTTPException:
                v15, pred15, v30, pred30 = None, None, None, None
        payload = []
        for index, (zone_id, raw, record) in enumerate(kept):
            name, borough = names.get(zone_id, (f"Zone {zone_id}", ""))
            payload.append(
                {
                    "zone_id": zone_id,
                    "name": name,
                    "borough": borough,
                    "prediction_15": None if pred15 is None else float(pred15[index]),
                    "prediction_30": None if pred30 is None else float(pred30[index]),
                    "pu_15m": record["pu_15m"],
                    "pu_60m": record["pu_60m"],
                }
            )
        meta = _meta()
        event_time = None
        if meta.get("watermark_epoch"):
            event_time = datetime.fromtimestamp(meta["watermark_epoch"], tz=NY).isoformat()
        return {
            "model_15": v15,
            "model_30": v30,
            "event_time": event_time,
            "watermark_epoch": meta.get("watermark_epoch"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "zones": payload,
        }

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    port = int(os.environ.get("PORT", "8841"))
    metrics_port = int(os.environ.get("METRICS_PORT", "8841"))
    if metrics_port != port:
        serve_metrics(metrics_port)
    # When the metrics port equals the API port, Prometheus scrapes /metrics on the app.
    app = create_app()
    if metrics_port == port:
        from prometheus_client import make_asgi_app

        app.mount("/metrics", make_asgi_app())
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
