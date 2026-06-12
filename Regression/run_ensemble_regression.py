# Module: run_ensemble_regression
# Purpose: Train stacking and weighted-blend ensembles for target_price_eur.
# Usage: py Regression/run_ensemble_regression.py [--device auto|gpu|cpu] [--target-mode log_price_eur]

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split

REG_DIR = Path(__file__).resolve().parent
if str(REG_DIR) not in sys.path:
    sys.path.insert(0, str(REG_DIR))

from base_models import build_base_estimators  # noqa: E402
from config import (  # noqa: E402
    CONTENT_DIR,
    CV_FOLDS,
    LEAKAGE_COLS,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    PRICE_MODEL_CSV,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from evaluation import evaluate_regression  # noqa: E402
from gpu import resolve_device  # noqa: E402


# Create output folders for metrics, models, and plots.
def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, MODEL_DIR, PLOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


# Load encoded price-prediction features and drop leakage-prone columns if present.
def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    path = CONTENT_DIR / PRICE_MODEL_CSV
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in {path.name}")

    leak_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leak_present:
        df = df.drop(columns=leak_present)

    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])
    return X, y


# Convert EUR targets to log space when training on skewed prices.
def transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log_price_eur":
        return np.log1p(y)
    if mode == "price_eur":
        return y
    raise ValueError("target_mode must be price_eur or log_price_eur")


# Map model predictions back to EUR for evaluation and blending.
def inverse_transform(pred: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log_price_eur":
        return np.expm1(pred)
    return pred


# Package one holdout metric row for a single model or ensemble.
def evaluate_holdout(
    name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    baseline: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"model": name, **evaluate_regression(y_test, y_pred, baseline)}
    if extra:
        row.update(extra)
    return row


# Out-of-fold MAE in EUR; used to weight base models in the blend.
def cv_mae_eur(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    target_mode: str,
    folds: int,
) -> float:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    y_train = transform_target(y, target_mode)

    for train_idx, val_idx in kfold.split(X):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y_train[train_idx])
        pred = est.predict(X.iloc[val_idx])
        oof[val_idx] = inverse_transform(pred, target_mode)

    return float(np.mean(np.abs(y - oof)))


# Fit each base model and compute inverse-CV-MAE blend weights.
def fit_weighted_blend(
    estimators: list[tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_mode: str,
    folds: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    weights: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    y_t = transform_target(y_train, target_mode)

    for name, est in estimators:
        cv_mae = cv_mae_eur(est, X_train, y_train, target_mode, folds)
        weights[name] = 1.0 / max(cv_mae, 1e-6)
        model = clone(est)
        model.fit(X_train, y_t)
        fitted[name] = model

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights, fitted


# Weighted average of fitted base-model predictions in EUR space.
def predict_weighted_blend(
    fitted: dict[str, Any],
    weights: dict[str, float],
    X: pd.DataFrame,
    target_mode: str,
) -> np.ndarray:
    pred = np.zeros(len(X))
    for name, model in fitted.items():
        p = inverse_transform(model.predict(X), target_mode)
        pred += weights[name] * p
    return pred


# Bar chart of holdout MAE for all trained models and ensembles.
def plot_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    plot_df = metrics.sort_values("MAE").copy()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="MAE", hue="ensemble_type", dodge=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("Holdout MAE by model (EUR)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# CLI entry: train base models, stacking, weighted blend, and save artifacts.
def main() -> None:
    parser = argparse.ArgumentParser(description="GSM price ensemble regression")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument(
        "--target-mode",
        choices=["price_eur", "log_price_eur"],
        default="log_price_eur",
        help="log_price_eur usually works best on skewed prices",
    )
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    ensure_dirs()
    plan = resolve_device(args.device)
    print("[device]", plan)
    for note in plan.notes:
        print(" ", note)

    X, y = load_dataset()
    print(f"[data] {PRICE_MODEL_CSV}: {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.to_numpy(), test_size=args.test_size, random_state=RANDOM_STATE
    )
    baseline_test = np.full(len(y_test), np.median(y_train))

    base_estimators = build_base_estimators(plan)
    print("[base models]", [name for name, _ in base_estimators])

    rows: list[dict[str, Any]] = []
    fitted_singletons: dict[str, Any] = {}
    y_train_t = transform_target(y_train, args.target_mode)

    for name, est in base_estimators:
        t0 = time.time()
        model = clone(est)
        model.fit(X_train, y_train_t)
        pred = inverse_transform(model.predict(X_test), args.target_mode)
        rows.append(
            evaluate_holdout(
                name,
                y_test,
                pred,
                baseline_test,
                {"ensemble_type": "base", "fit_seconds": time.time() - t0, "target_mode": args.target_mode},
            )
        )
        fitted_singletons[name] = model

    stack_estimators = [(n, clone(e)) for n, e in base_estimators]
    stack = StackingRegressor(
        estimators=stack_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
        cv=args.cv_folds,
        n_jobs=-1,
        passthrough=False,
    )
    t0 = time.time()
    stack.fit(X_train, y_train_t)
    stack_pred = inverse_transform(stack.predict(X_test), args.target_mode)
    rows.append(
        evaluate_holdout(
            "stacking_ridge_meta",
            y_test,
            stack_pred,
            baseline_test,
            {
                "ensemble_type": "stacking",
                "fit_seconds": time.time() - t0,
                "target_mode": args.target_mode,
            },
        )
    )

    t0 = time.time()
    blend_weights, blend_fitted = fit_weighted_blend(
        base_estimators, X_train, y_train, args.target_mode, args.cv_folds
    )
    blend_pred = predict_weighted_blend(blend_fitted, blend_weights, X_test, args.target_mode)
    rows.append(
        evaluate_holdout(
            "weighted_blend_cv",
            y_test,
            blend_pred,
            baseline_test,
            {
                "ensemble_type": "weighted_blend",
                "fit_seconds": time.time() - t0,
                "target_mode": args.target_mode,
            },
        )
    )

    metrics = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "ensemble_holdout_metrics.csv", index=False)

    pred_out = pd.DataFrame(
        {
            "y_true_eur": y_test,
            "stacking_pred_eur": stack_pred,
            "weighted_blend_pred_eur": blend_pred,
        }
    )
    pred_out.to_csv(OUTPUT_DIR / "ensemble_holdout_predictions.csv", index=False)

    with open(OUTPUT_DIR / "blend_weights.json", "w", encoding="utf-8") as f:
        json.dump(blend_weights, f, indent=2)

    package = {
        "target_mode": args.target_mode,
        "device_plan": plan.__dict__,
        "blend_weights": blend_weights,
        "blend_models": blend_fitted,
        "stacking_model": stack,
        "feature_names": X.columns.tolist(),
        "train_median_target": float(np.median(y_train)),
    }
    joblib.dump(package, MODEL_DIR / "ensemble_price_regressor.pkl")

    plot_metrics(metrics, PLOT_DIR / "ensemble_holdout_mae.png")

    meta = {
        "device_plan": plan.__dict__,
        "target_mode": args.target_mode,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "base_models": [n for n, _ in base_estimators],
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[holdout metrics]")
    print(metrics[["ensemble_type", "model", "MAE", "RMSE", "R2", "fit_seconds"]].to_string(index=False))
    print(f"\n[best] {metrics.iloc[0]['model']}  MAE={metrics.iloc[0]['MAE']:.2f} EUR")
    print(f"[saved] {OUTPUT_DIR / 'ensemble_holdout_metrics.csv'}")
    print(f"[saved] {MODEL_DIR / 'ensemble_price_regressor.pkl'}")


if __name__ == "__main__":
    main()
