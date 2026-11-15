from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from surgecast.features.compute import FEATURE_COLUMNS
from surgecast.serving.app import ChampionCache, create_app
from surgecast.training.train import fit_lightgbm


def _training_frame(shift: float) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    pu = np.linspace(0, 40, 60)
    data = {"zone_id": np.full(60, 7, dtype=np.int32)}
    for column in FEATURE_COLUMNS:
        if column == "zone_id":
            continue
        if column == "pu_15m":
            data[column] = pu
        else:
            data[column] = np.ones(60)
    frame = pd.DataFrame(data)
    y = pu + shift
    cut = 50
    return frame.iloc[:cut], y[:cut], frame.iloc[cut:], y[cut:]


def _booster(shift: float):
    train_x, train_y, val_x, val_y = _training_frame(shift)
    return fit_lightgbm(
        train_x,
        train_y,
        val_x,
        val_y,
        params={"min_data_in_leaf": 1, "num_leaves": 15, "learning_rate": 0.2},
        num_boost_round=40,
        early_stopping_rounds=10,
    )


def _hash(pu_15m: float) -> dict[str, str]:
    raw = {"zone_id": "7", "bucket": "20", "decision_epoch": "1700000000"}
    for column in FEATURE_COLUMNS:
        if column == "zone_id":
            continue
        raw[column] = "1" if column != "pu_15m" else str(pu_15m)
    return raw


def test_predict_uses_redis_features_and_hot_swaps_model():
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    client.hset("zone:7", mapping=_hash(20))
    low = _booster(0.0)
    high = _booster(30.0)
    cache = ChampionCache(
        loader=lambda: (1, low, 1, low),
        versions=lambda: (1, 1),
    )
    cache.model_15 = low
    cache.model_30 = low
    cache.version_15 = 1
    cache.version_30 = 1
    app = create_app(redis_client=client, cache=cache, start_background=False, zone_table={7: ("Midtown", "Manhattan")})
    with TestClient(app) as http:
        health = http.get("/api/health")
        assert health.status_code == 200
        assert health.json()["model_15"] == 1
        first = http.get("/api/predict/7?horizon=15")
        assert first.status_code == 200
        body = first.json()
        assert body["name"] == "Midtown"
        assert body["prediction"] == pytest.approx(20, abs=2.0)
        missing = http.get("/api/predict/8")
        assert missing.status_code == 404
        bad = http.get("/api/predict/7?horizon=5")
        assert bad.status_code == 422
        cache.loader = lambda: (2, high, 2, high)
        cache.versions = lambda: (2, 2)
        reloaded = http.post("/api/admin/reload")
        assert reloaded.status_code == 200
        assert reloaded.json()["model_15"] == 2
        second = http.get("/api/predict/7?horizon=15")
        assert second.status_code == 200
        assert second.json()["prediction"] == pytest.approx(50, abs=3.0)
        zones = http.get("/api/zones")
        assert zones.status_code == 200
        assert zones.json()["zones"][0]["zone_id"] == 7
        assert zones.json()["zones"][0]["prediction_15"] == pytest.approx(50, abs=3.0)
