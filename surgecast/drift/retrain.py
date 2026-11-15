"""Retrain LightGBM on the live bucket counts and promote a new champion."""

from __future__ import annotations

import time

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from surgecast.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MODEL_NAME_15,
    MODEL_NAME_30,
    ORIGIN_EPOCH,
    SCHEMA_VERSION,
)
from surgecast.features.compute import FEATURE_COLUMNS
from surgecast.serving.model_loader import use_tracking_uri
from surgecast.training.dataset import build_frame, matrix
from surgecast.training.metrics import regression_scores
from surgecast.training.train import LGB_PARAMS, fit_lightgbm


def _register(booster, name: str, example: pd.DataFrame, scores: dict, reason: str) -> int:
    with mlflow.start_run(run_name=f"retrain-{name}"):
        mlflow.log_params({key: value for key, value in LGB_PARAMS.items()})
        mlflow.log_param("reason", reason[:200])
        flat = {
            key: float(value)
            for key, value in scores.items()
            if isinstance(value, (int, float)) and value is not None
        }
        if flat:
            mlflow.log_metrics(flat)
        mlflow.set_tag("feature_schema_version", str(SCHEMA_VERSION))
        mlflow.set_tag("role", "retrain")
        mlflow.log_dict({"columns": FEATURE_COLUMNS, "schema_version": SCHEMA_VERSION}, "feature_columns.json")
        info = mlflow.lightgbm.log_model(
            booster,
            artifact_path="model",
            registered_model_name=name,
            input_example=example.head(2),
        )
    version = int(info.registered_model_version)
    MlflowClient().set_registered_model_alias(name, CHAMPION_ALIAS, str(version))
    return version


def retrain_from_counts(counts, reason: str, origin_epoch: int = ORIGIN_EPOCH, num_boost_round: int = 200) -> dict:
    """Fit both horizons on ``counts`` and point the champion aliases at them.

    The last 12 hours of decision buckets are the early-stopping slice. The
    rest, after the usual one-week warmup inside ``build_frame``, is the train set.
    """
    started = time.perf_counter()
    use_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    frame = build_frame(counts, origin_epoch=origin_epoch)
    if len(frame) < 500:
        raise RuntimeError(f"not enough labeled rows to retrain ({len(frame)})")
    max_bucket = int(frame["bucket"].max())
    val_start = max_bucket - 12 * 12
    train_df = frame.loc[frame["bucket"] < val_start]
    val_df = frame.loc[frame["bucket"] >= val_start]
    if len(train_df) < 200 or len(val_df) < 50:
        cut = max(1, int(len(frame) * 0.9))
        train_df = frame.iloc[:cut]
        val_df = frame.iloc[cut:]
    train_x, y15, y30 = matrix(train_df)
    val_x, v15, v30 = matrix(val_df)
    model_15 = fit_lightgbm(train_x, y15, val_x, v15, num_boost_round=num_boost_round)
    model_30 = fit_lightgbm(train_x, y30, val_x, v30, num_boost_round=num_boost_round)
    hold_x, h15, h30 = matrix(val_df)
    scores_15 = regression_scores(h15, model_15.predict(hold_x))
    scores_30 = regression_scores(h30, model_30.predict(hold_x))
    version_15 = _register(model_15, MODEL_NAME_15, train_x, scores_15, reason)
    version_30 = _register(model_30, MODEL_NAME_30, train_x, scores_30, reason)
    return {
        "version_15": version_15,
        "version_30": version_30,
        "rows": int(len(frame)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "val_scores_15": scores_15,
        "val_scores_30": scores_30,
        "seconds": time.perf_counter() - started,
        "reason": reason,
    }
