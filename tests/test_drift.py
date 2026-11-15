from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from surgecast.drift.detect import DriftReport, detect_drift, should_retrain


def _frame(scale_today: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seasonal_15 = rng.poisson(8, 500).astype(float) + 1.0
    seasonal_30 = seasonal_15 * 2
    return pd.DataFrame(
        {
            "pu_15m": seasonal_15 * scale_today,
            "pu_30m": seasonal_30 * scale_today,
            "seasonal_yday_15": seasonal_15,
            "seasonal_yday_30": seasonal_30,
        }
    )


def test_identical_sample_is_not_drift():
    reference = _frame(1.0, seed=1)
    report = detect_drift(reference, reference.copy())
    assert report.dataset_drift is False
    assert report.share_drifted == 0.0


def test_scaled_demand_is_drift():
    reference = _frame(1.0, seed=2)
    current = _frame(3.0, seed=2)
    report = detect_drift(reference, current)
    assert report.dataset_drift is True
    assert report.share_drifted >= 0.5


def test_should_retrain_respects_flag_and_cooldown():
    drifted = DriftReport(True, 0.8, 10, 10, 0.01)
    stable = DriftReport(False, 0.0, 10, 10, 0.01)
    assert should_retrain(drifted, None, now=100.0, cooldown_sec=50.0) is True
    assert should_retrain(drifted, last_retrain_at=80.0, now=100.0, cooldown_sec=50.0) is False
    assert should_retrain(drifted, last_retrain_at=40.0, now=100.0, cooldown_sec=50.0) is True
    assert should_retrain(stable, None, now=100.0, cooldown_sec=50.0) is False


def test_retrain_promotes_a_new_champion(tmp_path, monkeypatch):
    import mlflow
    from surgecast.drift.retrain import retrain_from_counts
    from surgecast.serving.model_loader import champion_versions

    uri = f"file:{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    n_buckets = 2016 + 700
    counts = np.zeros((264, n_buckets), dtype=np.int32)
    for zone in range(1, 16):
        counts[zone, :] = zone
        counts[zone, 2400:] = zone * 3
    first = retrain_from_counts(counts, reason="test-initial", num_boost_round=20)
    second = retrain_from_counts(counts, reason="test-drift", num_boost_round=20)
    assert second["version_15"] > first["version_15"]
    assert second["version_30"] > first["version_30"]
    versions = champion_versions(uri)
    assert versions == (second["version_15"], second["version_30"])
    pytest.importorskip("lightgbm")
