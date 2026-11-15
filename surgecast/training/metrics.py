"""Regression metrics for count forecasts.

MAPE is only defined for rows with an actual count above zero. MAE and RMSE are
reported twice: once on every row, and once on the positive rows. WAPE
(sum of absolute errors divided by sum of actuals, on positive rows) is an
additional scale-free number. It is not a substitute for MAPE.
"""

from __future__ import annotations

import numpy as np


def clip_prediction(yhat: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(yhat, dtype=np.float64), 0.0)


def regression_scores(y: np.ndarray, yhat: np.ndarray) -> dict[str, float | int | None]:
    actual = np.asarray(y, dtype=np.float64)
    pred = clip_prediction(yhat)
    if actual.shape != pred.shape:
        raise ValueError(f"shape mismatch: y {actual.shape} yhat {pred.shape}")
    err = pred - actual
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err)) if actual.size else None
    rmse = float(np.sqrt(np.mean(err**2))) if actual.size else None
    positive = actual > 0
    n_pos = int(positive.sum())
    if n_pos:
        mape = float(np.mean(abs_err[positive] / actual[positive]) * 100.0)
        mae_pos = float(np.mean(abs_err[positive]))
        rmse_pos = float(np.sqrt(np.mean(err[positive] ** 2)))
        denom = float(np.sum(actual[positive]))
        wape = float(np.sum(abs_err[positive]) / denom * 100.0) if denom else None
        mean_y = float(np.mean(actual[positive]))
    else:
        mape = mae_pos = rmse_pos = wape = mean_y = None
    return {
        "n": int(actual.size),
        "n_positive": n_pos,
        "mae": mae,
        "rmse": rmse,
        "mae_y_gt_0": mae_pos,
        "rmse_y_gt_0": rmse_pos,
        "mape_pct_y_gt_0": mape,
        "wape_pct_y_gt_0": wape,
        "mean_y_positive": mean_y,
    }
