"""Regression evaluation metrics (aligned with modeling/run_modeling.py)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    score = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(score) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    if baseline_mae == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred) / baseline_mae)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1.0, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, baseline_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }
