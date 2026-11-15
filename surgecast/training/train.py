"""Train LightGBM demand models, score naive baselines, optionally a small MLP.

The served models are the LightGBM boosters registered in MLflow under the
``champion`` alias. The MLP is logged as a comparison run and is not promoted.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from surgecast.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MODEL_NAME_15,
    MODEL_NAME_30,
    PROCESSED_DIR,
    RESULTS_DIR,
    SCHEMA_VERSION,
    TEST_START_DT,
    VAL_START_DT,
    ensure_dirs,
)
from surgecast.data.prepare import load_counts
from surgecast.features.compute import DRIFT_COLUMNS, FEATURE_COLUMNS
from surgecast.serving.model_loader import use_tracking_uri
from surgecast.training.dataset import build_frame, matrix, time_splits
from surgecast.training.metrics import clip_prediction, regression_scores

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 7,
    "num_threads": 4,
    "force_col_wise": True,
}


def fit_lightgbm(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    params: dict | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
) -> lgb.Booster:
    used = dict(LGB_PARAMS)
    if params:
        used.update(params)
    train_set = lgb.Dataset(
        train_x,
        label=train_y,
        categorical_feature=["zone_id"],
        free_raw_data=False,
    )
    valid_sets = [train_set]
    callbacks = [lgb.log_evaluation(period=50)]
    if len(val_x):
        valid_sets.append(
            lgb.Dataset(val_x, label=val_y, categorical_feature=["zone_id"], reference=train_set, free_raw_data=False)
        )
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    return lgb.train(
        used,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _baseline_block(test: pd.DataFrame, horizon: str) -> dict:
    if horizon == "15":
        y = test["y15"].to_numpy()
        baselines = {
            "last_value": test["pu_15m"].to_numpy(),
            "seasonal_naive_yesterday": test["seasonal_yday_15"].to_numpy(),
        }
    else:
        y = test["y30"].to_numpy()
        baselines = {
            "last_value": test["pu_30m"].to_numpy(),
            "seasonal_naive_yesterday": test["seasonal_yday_30"].to_numpy(),
        }
    return {name: regression_scores(y, pred) for name, pred in baselines.items()}


def _register(booster: lgb.Booster, name: str, example: pd.DataFrame, metrics: dict) -> int:
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: v for k, v in LGB_PARAMS.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None})
        mlflow.set_tag("feature_schema_version", str(SCHEMA_VERSION))
        mlflow.set_tag("role", "champion-candidate")
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


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def train(skip_torch: bool = False, num_boost_round: int = 300) -> dict:
    ensure_dirs()
    print("loading trips", flush=True)
    started = time.perf_counter()
    counts, meta = load_counts()
    print(f"counts {counts.shape} kept {meta.get('n_kept')} in {time.perf_counter() - started:.1f}s", flush=True)
    frame = build_frame(counts)
    print(f"frame rows {len(frame)}", flush=True)
    train_df, val_df, test_df = time_splits(frame)
    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"empty split: train={len(train_df)} val={len(val_df)} test={len(test_df)} rows={len(frame)}"
        )
    train_x, train_y15, train_y30 = matrix(train_df)
    val_x, val_y15, val_y30 = matrix(val_df)
    test_x, test_y15, test_y30 = matrix(test_df)

    use_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)

    t0 = time.perf_counter()
    model_15 = fit_lightgbm(train_x, train_y15, val_x, val_y15, num_boost_round=num_boost_round)
    pred_15 = clip_prediction(model_15.predict(test_x))
    scores_15 = regression_scores(test_y15, pred_15)
    t15 = time.perf_counter() - t0

    t0 = time.perf_counter()
    model_30 = fit_lightgbm(train_x, train_y30, val_x, val_y30, num_boost_round=num_boost_round)
    pred_30 = clip_prediction(model_30.predict(test_x))
    scores_30 = regression_scores(test_y30, pred_30)
    t30 = time.perf_counter() - t0

    version_15 = _register(model_15, MODEL_NAME_15, train_x, scores_15)
    version_30 = _register(model_30, MODEL_NAME_30, train_x, scores_30)

    reference = train_df.loc[:, DRIFT_COLUMNS]
    if len(reference) > 20_000:
        reference = reference.sample(20_000, random_state=7)
    reference_path = PROCESSED_DIR / "reference_features.parquet"
    reference.to_parquet(reference_path, index=False)

    torch_report: dict | None = None
    torch_error = None
    if not skip_torch:
        try:
            from surgecast.training.torch_model import train_mlp

            t0 = time.perf_counter()
            torch_15 = train_mlp(train_df, val_df, test_df, "15")
            torch_30 = train_mlp(train_df, val_df, test_df, "30")
            torch_seconds = time.perf_counter() - t0
            with mlflow.start_run(run_name="torch-mlp-comparison"):
                mlflow.log_param("epochs", 6)
                mlflow.log_param("hidden", "64,32")
                mlflow.log_metrics(
                    {
                        "mae_y_gt_0_15": torch_15["scores"]["mae_y_gt_0"],
                        "mae_y_gt_0_30": torch_30["scores"]["mae_y_gt_0"],
                        "mape_15": torch_15["scores"]["mape_pct_y_gt_0"],
                        "mape_30": torch_30["scores"]["mape_pct_y_gt_0"],
                    }
                )
                mlflow.set_tag("role", "comparison-not-served")
            torch_report = {"seconds": torch_seconds, "horizon_15": torch_15, "horizon_30": torch_30}
        except Exception as exc:  # recorded, not hidden
            torch_error = f"{type(exc).__name__}: {exc}"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "python -m surgecast.training.train",
        "hardware": _hardware(),
        "mlflow_tracking_uri": use_tracking_uri(),
        "counts_meta": meta,
        "rows": {
            "frame": int(len(frame)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "splits": {
            "train_min_bucket": int(train_df["bucket"].min()),
            "val_start_local": VAL_START_DT.isoformat(),
            "test_start_local": TEST_START_DT.isoformat(),
        },
        "lightgbm": {
            "params": LGB_PARAMS,
            "num_boost_round_cap": num_boost_round,
            "best_iteration_15": int(model_15.best_iteration or num_boost_round),
            "best_iteration_30": int(model_30.best_iteration or num_boost_round),
            "fit_seconds_15": t15,
            "fit_seconds_30": t30,
            "version_15": version_15,
            "version_30": version_30,
            "horizon_15": {
                "model": scores_15,
                "baselines": _baseline_block(test_df, "15"),
            },
            "horizon_30": {
                "model": scores_30,
                "baselines": _baseline_block(test_df, "30"),
            },
        },
        "torch": torch_report,
        "torch_error": torch_error,
        "reference_features": str(reference_path),
        "reference_rows": int(len(reference)),
        "total_seconds": time.perf_counter() - started,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "train_metrics.json"
    out.write_text(json.dumps(report, indent=2, default=_json_default))
    print(
        json.dumps(
            {
                "wrote": str(out),
                "version_15": version_15,
                "version_30": version_30,
                "mae15": scores_15["mae_y_gt_0"],
                "mae30": scores_30["mae_y_gt_0"],
            }
        )
    )
    return report


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Surgecast demand models")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--num-boost-round", type=int, default=300)
    args = parser.parse_args()
    train(skip_torch=args.skip_torch, num_boost_round=args.num_boost_round)


if __name__ == "__main__":
    main()
