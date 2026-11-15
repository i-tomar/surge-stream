"""Evidently data-drift check used by the monitor and by tests.

The monitored values are ratios of recent pickups to the same window yesterday.
A pure time-of-day swing moves the numerator and the seasonal feature together,
so the ratio stays near 1. A demand shock moves the numerator first and the
ratio shifts. That is the signal that triggers retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from surgecast.config import DRIFT_PSI_THRESHOLD, DRIFT_SHARE_THRESHOLD

SOURCE_COLUMNS = ["pu_15m", "pu_30m", "seasonal_yday_15", "seasonal_yday_30"]


@dataclass(frozen=True)
class DriftReport:
    dataset_drift: bool
    share_drifted: float
    n_reference: int
    n_current: int
    seconds: float


def drift_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"drift frame missing columns: {missing}")
    source = frame.loc[:, SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    source = source.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return pd.DataFrame(
        {
            "ratio_yday_15": source["pu_15m"] / (source["seasonal_yday_15"] + 1.0),
            "ratio_yday_30": source["pu_30m"] / (source["seasonal_yday_30"] + 1.0),
        }
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    share_threshold: float = DRIFT_SHARE_THRESHOLD,
    psi_threshold: float = DRIFT_PSI_THRESHOLD,
) -> DriftReport:
    """Flag drift when at least half of the ratio features have PSI above 0.2."""
    ref = drift_matrix(reference)
    cur = drift_matrix(current)
    if len(ref) > 8000:
        ref = ref.sample(8000, random_state=0)
    if len(cur) > 8000:
        cur = cur.sample(8000, random_state=0)
    if len(ref) < 50 or len(cur) < 50:
        return DriftReport(False, 0.0, int(len(ref)), int(len(cur)), 0.0)
    started = time.perf_counter()
    report = Report(
        metrics=[
            DataDriftPreset(
                num_stattest="psi",
                num_stattest_threshold=psi_threshold,
                drift_share=share_threshold,
            )
        ]
    )
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()["metrics"][0]["result"]
    return DriftReport(
        dataset_drift=bool(result["dataset_drift"]),
        share_drifted=float(result["share_of_drifted_columns"]),
        n_reference=int(len(ref)),
        n_current=int(len(cur)),
        seconds=time.perf_counter() - started,
    )


def should_retrain(
    report: DriftReport,
    last_retrain_at: float | None,
    now: float,
    cooldown_sec: float,
) -> bool:
    if not report.dataset_drift:
        return False
    if last_retrain_at is not None and (now - last_retrain_at) < cooldown_sec:
        return False
    return True
