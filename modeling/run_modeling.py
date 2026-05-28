"""DS Team 7 GSM smartphone modeling pipeline.

This script is intentionally self-contained so the submitted notebook can run it
in Colab or locally and reproduce every modeling output.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback only for restricted envs.
    from sklearn.ensemble import HistGradientBoostingRegressor

    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELING_DIR = ROOT_DIR / "modeling"
OUTPUT_DIR = MODELING_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = MODELING_DIR / "models"
DOC_DIR = MODELING_DIR / "docs"

CONTENT_DIR = ROOT_DIR / "content"

PRICE_MODEL_CSV = "gsm_processed_all(price_prediction).csv"
PRICE_RAW_CSV = "gsm_processed(price_prediction).csv"
RECO_MODEL_CSV = "gsm_processed_all(recommendation).csv"
RECO_RAW_CSV = "gsm_processed(recommendation).csv"

TARGET_COL = "target_price_eur"


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline."""
    for path in [OUTPUT_DIR, PLOT_DIR, MODEL_DIR, DOC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def find_data_source() -> Path:
    """Locate the team preprocessing output directory in the GitHub repo layout."""
    required = [PRICE_MODEL_CSV, PRICE_RAW_CSV, RECO_MODEL_CSV, RECO_RAW_CSV]
    if all((CONTENT_DIR / member).exists() for member in required):
        return CONTENT_DIR
    missing = [member for member in required if not (CONTENT_DIR / member).exists()]
    raise FileNotFoundError(
        "GitHub repo layout 기준 입력 CSV를 찾을 수 없습니다.\n"
        f"프로젝트 루트: {ROOT_DIR}\n"
        f"필수 폴더: {CONTENT_DIR}\n"
        "누락 파일: " + ", ".join(missing) + "\n"
        "기대 구조: DS_Team7/content/*.csv 와 DS_Team7/modeling/run_modeling.py"
    )


def read_modeling_csv(data_source: Path, member_name: str) -> pd.DataFrame:
    """Read one modeling CSV from DS_Team7/content."""
    return pd.read_csv(data_source / member_name)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent."""
    denom = np.abs(y_true) + np.abs(y_pred)
    score = np.where(denom == 0, 0, 2 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(score) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    """MASE against a median-price naive baseline."""
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    if baseline_mae == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred) / baseline_mae)


def evaluate_price_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> dict[str, float]:
    """Return all evaluation metrics in original EUR units."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, baseline_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_lgbm() -> Any:
    """Create the tree boosting model used for the strongest tabular baseline."""
    if LIGHTGBM_AVAILABLE:
        return LGBMRegressor(
            n_estimators=450,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=450,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )


def make_models() -> dict[str, Any]:
    """Define a compact model set that runs quickly in Colab CPU."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM" if LIGHTGBM_AVAILABLE else "HistGradientBoosting": make_lgbm(),
    }


@dataclass
class RegressionResult:
    metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    best_model: Any
    best_row: pd.Series
    feature_names: list[str]
    feature_medians: dict[str, float]


def train_and_evaluate_regression(price_model_df: pd.DataFrame) -> RegressionResult:
    """Train price prediction models and store holdout/CV metrics."""
    if TARGET_COL not in price_model_df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 회귀 입력에 없습니다.")

    X = price_model_df.drop(columns=[TARGET_COL]).copy()
    y = price_model_df[TARGET_COL].astype(float).copy()
    feature_names = X.columns.tolist()
    feature_medians = X.median(numeric_only=True).to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    baseline_test = np.full(len(y_test), y_train.median())

    holdout_rows: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, str], Any] = {}
    models = make_models()

    for target_mode in ["price_eur", "log_price_eur"]:
        train_target = np.log1p(y_train) if target_mode == "log_price_eur" else y_train
        for model_name, model in models.items():
            start = time.time()
            model.fit(X_train, train_target)
            fit_seconds = time.time() - start
            pred = model.predict(X_test)
            if target_mode == "log_price_eur":
                pred = np.expm1(pred)
            row = {
                "target_mode": target_mode,
                "model": model_name,
                **evaluate_price_metrics(y_test.to_numpy(), pred, baseline_test),
                "fit_seconds": fit_seconds,
            }
            holdout_rows.append(row)
            fitted_models[(target_mode, model_name)] = model

    holdout_metrics = pd.DataFrame(holdout_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    holdout_metrics.to_csv(OUTPUT_DIR / "regression_holdout_metrics.csv", index=False)
    holdout_metrics.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)

    cv_rows: list[dict[str, Any]] = []
    cv_fold_rows: list[dict[str, Any]] = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for target_mode in ["price_eur", "log_price_eur"]:
        for model_name, base_model in make_models().items():
            fold_metrics: list[dict[str, float]] = []
            start = time.time()
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                model = make_models()[model_name]
                train_target = np.log1p(y_tr) if target_mode == "log_price_eur" else y_tr
                model.fit(X_tr, train_target)
                pred = model.predict(X_te)
                if target_mode == "log_price_eur":
                    pred = np.expm1(pred)
                baseline = np.full(len(y_te), y_tr.median())
                metrics = evaluate_price_metrics(y_te.to_numpy(), pred, baseline)
                fold_metrics.append(metrics)
                cv_fold_rows.append(
                    {
                        "target_mode": target_mode,
                        "model": model_name,
                        "fold": fold,
                        **metrics,
                    }
                )
            elapsed = time.time() - start
            avg_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            cv_rows.append(
                {
                    "target_mode": target_mode,
                    "model": model_name,
                    **avg_metrics,
                    "fit_seconds": elapsed,
                }
            )

    cv_metrics = pd.DataFrame(cv_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    cv_metrics.to_csv(OUTPUT_DIR / "regression_cv_metrics.csv", index=False)
    cv_metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    pd.DataFrame(cv_fold_rows).to_csv(OUTPUT_DIR / "regression_cv_fold_metrics.csv", index=False)

    best_row = holdout_metrics.iloc[0]
    best_key = (str(best_row["target_mode"]), str(best_row["model"]))
    best_model = fitted_models[best_key]

    package = {
        "model": best_model,
        "target_mode": best_key[0],
        "model_name": best_key[1],
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
    }
    joblib.dump(package, MODEL_DIR / "best_price_regressor.pkl")

    return RegressionResult(holdout_metrics, cv_metrics, best_model, best_row, feature_names, feature_medians)


def plot_regression_metrics(cv_metrics: pd.DataFrame) -> None:
    """Save quick CV metric comparison plots."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=cv_metrics, x="model", y="MAE", hue="target_mode")
    plt.title("5-fold CV MAE by model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "regression_cv_mae.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=cv_metrics, x="model", y="R2", hue="target_mode")
    plt.title("5-fold CV R2 by model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "regression_cv_r2.png", dpi=160)
    plt.close()


def feature_importance_analysis(price_model_df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate feature importance and brand-related contribution."""
    X = price_model_df.drop(columns=[TARGET_COL])
    y = price_model_df[TARGET_COL].astype(float)

    rf = RandomForestRegressor(
        n_estimators=260,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm = make_lgbm()
    rf.fit(X, y)
    lgbm.fit(X, y)

    rows: list[dict[str, Any]] = []
    rows.extend(
        {
            "model": "Random Forest (price_eur)",
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(feature_names, rf.feature_importances_)
    )

    if hasattr(lgbm, "booster_"):
        gain = lgbm.booster_.feature_importance(importance_type="gain")
    elif hasattr(lgbm, "feature_importances_"):
        gain = lgbm.feature_importances_
    else:
        gain = np.zeros(len(feature_names))
    rows.extend(
        {
            "model": "LightGBM_gain (price_eur)" if LIGHTGBM_AVAILABLE else "HistGradientBoosting (price_eur)",
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(feature_names, gain)
    )

    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values(["model", "importance"], ascending=[True, False])
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance_best_target.csv", index=False)

    top_features = []
    for model_name, group in importance_df.groupby("model"):
        top = group.sort_values("importance", ascending=False).head(15)
        top_features.append(top.assign(rank=range(1, len(top) + 1)))
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top, y="feature", x="importance")
        plt.title(f"Top 15 Feature Importance - {model_name}")
        plt.tight_layout()
        filename = "rf_feature_importance_top15.png" if model_name.startswith("Random") else "lgbm_feature_importance_top15.png"
        plt.savefig(PLOT_DIR / filename, dpi=160)
        plt.close()

    top_df = pd.concat(top_features, ignore_index=True)
    consensus = (
        top_df.groupby("feature")
        .agg(avg_rank=("rank", "mean"), model_count=("model", "nunique"))
        .query("model_count >= 2")
        .sort_values(["avg_rank", "feature"])
        .reset_index()
    )
    consensus.to_csv(OUTPUT_DIR / "consensus_price_factors.csv", index=False)

    brand_rows = []
    for model_name, group in importance_df.groupby("model"):
        total = group["importance"].sum()
        brand_sum = group.loc[group["feature"].str.startswith("cat__brand_group_"), "importance"].sum()
        brand_rows.append(
            {
                "model": model_name,
                "brand_importance_sum": brand_sum,
                "total_importance": total,
                "brand_importance_ratio": brand_sum / total if total else np.nan,
            }
        )
    brand_importance = pd.DataFrame(brand_rows)
    brand_importance.to_csv(OUTPUT_DIR / "brand_importance_summary.csv", index=False)

    return importance_df, consensus


def brand_premium_analysis(price_model_df: pd.DataFrame, price_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate raw and same-spec adjusted brand premiums."""
    brand_summary = (
        price_raw_df.groupby("brand_group")
        .agg(
            count=("price_eur", "size"),
            avg_price_eur=("price_eur", "mean"),
            median_price_eur=("price_eur", "median"),
            avg_is_premium_brand=("is_premium_brand", "mean"),
        )
        .sort_values("avg_price_eur", ascending=False)
        .reset_index()
    )
    brand_summary.to_csv(OUTPUT_DIR / "brand_premium_summary.csv", index=False)

    X = price_model_df.drop(columns=[TARGET_COL]).copy()
    y = price_model_df[TARGET_COL].astype(float)
    brand_cols = [c for c in X.columns if c.startswith("cat__brand_group_")]
    spec_cols = [c for c in X.columns if c not in brand_cols and c != "num__is_premium_brand"]
    spec_model = make_lgbm()
    spec_model.fit(X[spec_cols], y)
    pred = np.clip(spec_model.predict(X[spec_cols]), 1, None)

    premium_df = pd.DataFrame(
        {
            "brand_group": price_raw_df["brand_group"].values,
            "actual_price_eur": y.values,
            "spec_only_pred_price_eur": pred,
        }
    )
    premium_df["adjusted_brand_premium_eur"] = (
        premium_df["actual_price_eur"] - premium_df["spec_only_pred_price_eur"]
    )
    adjusted = (
        premium_df.groupby("brand_group")
        .agg(
            count=("actual_price_eur", "size"),
            avg_actual_price_eur=("actual_price_eur", "mean"),
            avg_spec_only_pred_price_eur=("spec_only_pred_price_eur", "mean"),
            avg_adjusted_brand_premium_eur=("adjusted_brand_premium_eur", "mean"),
            median_adjusted_brand_premium_eur=("adjusted_brand_premium_eur", "median"),
        )
        .reset_index()
    )
    adjusted["premium_ratio_vs_pred"] = (
        adjusted["avg_adjusted_brand_premium_eur"] / adjusted["avg_spec_only_pred_price_eur"]
    )
    adjusted = adjusted.sort_values("avg_adjusted_brand_premium_eur", ascending=False)
    adjusted.to_csv(OUTPUT_DIR / "adjusted_brand_premium_summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    plot_df = adjusted.head(12)
    sns.barplot(data=plot_df, x="avg_adjusted_brand_premium_eur", y="brand_group")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Adjusted Brand Premium by Brand Group")
    plt.xlabel("Actual price - spec-only predicted price (EUR)")
    plt.ylabel("Brand group")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "adjusted_brand_premium.png", dpi=160)
    plt.close()

    return adjusted


def cluster_segments(reco_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster recommendation rows into Entry, Mid-range, Flagship segments."""
    segment_features = [
        "price_eur",
        "spec_score_0_100",
        "ram_gb",
        "storage_gb",
        "battery_capacity_mah",
        "main_camera_max_mp",
        "ppi",
        "network_generation",
        "sensor_count",
    ]
    missing = sorted(set(segment_features) - set(reco_df.columns))
    if missing:
        raise ValueError(f"클러스터링 피처 누락: {missing}")

    cluster_input = reco_df[segment_features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(cluster_input)
    X_scaled = scaler.fit_transform(X_imputed)

    k_rows = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_scaled)
        k_rows.append(
            {
                "k": k,
                "inertia": KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled).inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            }
        )
    k_eval = pd.DataFrame(k_rows)
    k_eval.to_csv(OUTPUT_DIR / "cluster_k_evaluation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=k_eval, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow: KMeans inertia")
    sns.lineplot(data=k_eval, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].set_title("Silhouette by k")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_elbow_silhouette.png", dpi=160)
    plt.close()

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled)
    agg_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)
    comparison = pd.DataFrame(
        [
            {
                "algorithm": "KMeans",
                "n_samples": len(reco_df),
                "silhouette": silhouette_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            },
            {
                "algorithm": "Agglomerative_Ward",
                "n_samples": len(reco_df),
                "silhouette": silhouette_score(X_scaled, agg_labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, agg_labels),
            },
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "cluster_algorithm_comparison.csv", index=False)
    comparison.rename(columns={"silhouette": "value"}).head(1).to_csv(
        OUTPUT_DIR / "task_d_cluster_quality.csv", index=False
    )

    segmented = reco_df.copy()
    segmented["cluster"] = labels
    cluster_order = (
        segmented.groupby("cluster")
        .agg(avg_price_eur=("price_eur", "mean"), avg_spec_score_0_100=("spec_score_0_100", "mean"))
        .sort_values(["avg_price_eur", "avg_spec_score_0_100"])
        .reset_index()
    )
    label_map = dict(zip(cluster_order["cluster"], ["Entry", "Mid-range", "Flagship"]))
    segmented["segment"] = segmented["cluster"].map(label_map)

    segment_summary = (
        segmented.groupby("segment")
        .agg(
            count=("price_eur", "size"),
            avg_price_eur=("price_eur", "mean"),
            median_price_eur=("price_eur", "median"),
            avg_spec_score_0_100=("spec_score_0_100", "mean"),
            avg_ram_gb=("ram_gb", "mean"),
            avg_storage_gb=("storage_gb", "mean"),
            avg_battery_capacity_mah=("battery_capacity_mah", "mean"),
            avg_main_camera_max_mp=("main_camera_max_mp", "mean"),
            avg_ppi=("ppi", "mean"),
            avg_network_generation=("network_generation", "mean"),
        )
        .reset_index()
    )
    segment_summary["segment_order"] = segment_summary["segment"].map({"Entry": 0, "Mid-range": 1, "Flagship": 2})
    segment_summary = segment_summary.sort_values("segment_order").drop(columns=["segment_order"])
    segment_summary.to_csv(OUTPUT_DIR / "cluster_segment_summary.csv", index=False)
    segment_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    brand_dist = (
        segmented.groupby(["segment", "brand_group"]).size().reset_index(name="count")
        .sort_values(["segment", "count"], ascending=[True, False])
    )
    brand_dist["rank"] = brand_dist.groupby("segment")["count"].rank(method="first", ascending=False)
    brand_dist[brand_dist["rank"] <= 3].drop(columns=["rank"]).to_csv(
        OUTPUT_DIR / "cluster_top3_brand_distribution.csv", index=False
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": pca_xy[:, 0], "PC2": pca_xy[:, 1], "segment": segmented["segment"]})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="segment", s=25, alpha=0.75)
    plt.title("PCA view of smartphone segments")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_pca_segments.png", dpi=160)
    plt.close()

    segmented.to_csv(OUTPUT_DIR / "df_with_segments.csv", index=False)
    return segmented


def value_outliers_and_recommendations(segmented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect value-for-money outliers and build budget recommendations."""
    value_df = segmented.copy()
    value_df["performance_to_price_ratio"] = value_df["spec_score_0_100"] / value_df["price_eur"]
    value_df["segment_mean_value_score"] = value_df.groupby("segment")["value_score"].transform("mean")
    value_df["segment_median_price_eur"] = value_df.groupby("segment")["price_eur"].transform("median")
    value_df["value_score_lift_vs_segment"] = value_df["value_score"] / value_df["segment_mean_value_score"]
    value_df["price_discount_vs_segment_median"] = (
        value_df["segment_median_price_eur"] - value_df["price_eur"]
    ) / value_df["segment_median_price_eur"]

    market_features = [
        "spec_score_0_100",
        "ram_gb",
        "storage_gb",
        "battery_capacity_mah",
        "main_camera_max_mp",
        "ppi",
        "network_generation",
        "sensor_count",
        "phone_age",
    ]
    market_X = value_df[market_features].copy()
    market_y = value_df["price_eur"].astype(float)
    expected_price = np.zeros(len(value_df))
    market_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in market_kfold.split(market_X):
        market_model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=140,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
        market_model.fit(market_X.iloc[train_idx], market_y.iloc[train_idx])
        expected_price[test_idx] = market_model.predict(market_X.iloc[test_idx])
    value_df["expected_market_price_eur"] = np.clip(expected_price, 1, None)
    value_df["market_price_gap_eur"] = value_df["expected_market_price_eur"] - value_df["price_eur"]
    value_df["market_price_discount_ratio"] = (
        value_df["market_price_gap_eur"] / value_df["expected_market_price_eur"]
    )
    value_df["market_underpricing_z_in_segment"] = value_df.groupby("segment")[
        "market_price_discount_ratio"
    ].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    value_df["segment_median_performance_to_price_ratio"] = value_df.groupby("segment")[
        "performance_to_price_ratio"
    ].transform("median")
    value_df["market_underpriced_outlier"] = (
        (value_df["market_price_gap_eur"] > 0)
        & (value_df["market_underpricing_z_in_segment"] > 1.645)
        & (value_df["performance_to_price_ratio"] >= value_df["segment_median_performance_to_price_ratio"])
    )

    value_df["value_z_in_segment"] = value_df.groupby("segment")["value_score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    value_df["zscore_value_outlier"] = value_df["value_z_in_segment"] > 1.645

    iso_features = [
        "value_score",
        "performance_to_price_ratio",
        "market_price_discount_ratio",
        "spec_score_0_100",
        "price_eur",
        "phone_age",
    ]
    iso_input = value_df[iso_features].copy()
    iso_X = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(iso_input)
    isolation = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    value_df["isolation_raw_outlier"] = isolation.fit_predict(iso_X) == -1
    threshold = value_df["value_score"].quantile(0.75)
    value_df["isolation_value_outlier"] = (
        value_df["isolation_raw_outlier"] & (value_df["value_score"] >= threshold)
    )
    value_df["is_value_outlier"] = (
        value_df["zscore_value_outlier"]
        | value_df["isolation_value_outlier"]
        | value_df["market_underpriced_outlier"]
    )

    comparison = pd.DataFrame(
        [
            {"method": "Z-score within segment", "count": int(value_df["zscore_value_outlier"].sum())},
            {"method": "IsolationForest + high value_score", "count": int(value_df["isolation_value_outlier"].sum())},
            {
                "method": "Market expected-price underpricing Z-score",
                "count": int(value_df["market_underpriced_outlier"].sum()),
            },
            {
                "method": "Intersection",
                "count": int((value_df["zscore_value_outlier"] & value_df["isolation_value_outlier"]).sum()),
            },
            {"method": "Union(final)", "count": int(value_df["is_value_outlier"].sum())},
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "value_outlier_method_comparison.csv", index=False)

    validation = (
        value_df.groupby("is_value_outlier")
        .agg(
            count=("value_score", "size"),
            avg_price_eur=("price_eur", "mean"),
            avg_value_score=("value_score", "mean"),
            avg_performance_to_price_ratio=("performance_to_price_ratio", "mean"),
            avg_value_lift_vs_segment=("value_score_lift_vs_segment", "mean"),
            avg_price_discount_vs_segment_median=("price_discount_vs_segment_median", "mean"),
            avg_expected_market_price_eur=("expected_market_price_eur", "mean"),
            avg_market_price_gap_eur=("market_price_gap_eur", "mean"),
            avg_market_price_discount_ratio=("market_price_discount_ratio", "mean"),
        )
        .reset_index()
    )
    validation.to_csv(OUTPUT_DIR / "value_outlier_validation_summary.csv", index=False)

    value_outliers = value_df[value_df["is_value_outlier"]].sort_values(
        ["performance_to_price_ratio", "market_price_discount_ratio", "value_score"], ascending=False
    )
    value_outliers.to_csv(OUTPUT_DIR / "value_outliers.csv", index=False)
    value_outliers.to_csv(OUTPUT_DIR / "value_deal_outliers.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=value_df,
        x="price_eur",
        y="performance_to_price_ratio",
        hue="is_value_outlier",
        alpha=0.7,
        s=24,
    )
    plt.title("Value-for-money outliers")
    plt.ylabel("Performance-to-Price Ratio")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "value_outlier_scatter.png", dpi=160)
    plt.close()

    scenarios = [
        {"scenario": 1, "scenario_label": "학생 예산", "budget_eur": 200, "segment": None},
        {"scenario": 2, "scenario_label": "일반 직장인", "budget_eur": 400, "segment": "Mid-range"},
        {"scenario": 3, "scenario_label": "프리미엄 유저", "budget_eur": 800, "segment": "Flagship"},
        {"scenario": 4, "scenario_label": "입문자/세컨드폰", "budget_eur": 150, "segment": "Entry"},
    ]
    rec_frames = []
    validation_rows = []
    for scenario in scenarios:
        rec = recommend_phones(
            value_df,
            budget_eur=scenario["budget_eur"],
            top_n=10,
            segment_preference=scenario["segment"],
        )
        rec.insert(0, "scenario", scenario["scenario"])
        rec.insert(1, "scenario_label", scenario["scenario_label"])
        rec.insert(2, "budget_eur", scenario["budget_eur"])
        rec.to_csv(OUTPUT_DIR / f"recommendations_scenario_{scenario['scenario']}.csv", index=False)
        rec_frames.append(rec)
        validation_rows.append(
            {
                "scenario": scenario["scenario"],
                "scenario_label": scenario["scenario_label"],
                "budget_eur": scenario["budget_eur"],
                "row_count": len(rec),
                "max_price_eur": rec["price_eur"].max() if len(rec) else np.nan,
                "all_within_budget": bool((rec["price_eur"] <= scenario["budget_eur"]).all()) if len(rec) else True,
                "is_value_score_descending": bool(rec["value_score"].is_monotonic_decreasing) if len(rec) else True,
                "is_performance_to_price_ratio_descending": bool(
                    rec["performance_to_price_ratio"].is_monotonic_decreasing
                ) if len(rec) else True,
                "top_value_score": rec["value_score"].iloc[0] if len(rec) else np.nan,
                "top_performance_to_price_ratio": rec["performance_to_price_ratio"].iloc[0] if len(rec) else np.nan,
            }
        )
    recommendations = pd.concat(rec_frames, ignore_index=True)
    recommendations.to_csv(OUTPUT_DIR / "recommendations_all_scenarios.csv", index=False)
    recommendations.to_csv(OUTPUT_DIR / "budget_recommendations.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "recommendation_constraint_validation.csv", index=False)

    lift = (
        recommendations.groupby(["scenario", "scenario_label"])
        .agg(
            recommendation_mean_value_score=("value_score", "mean"),
            recommendation_mean_performance_to_price_ratio=("performance_to_price_ratio", "mean"),
        )
        .reset_index()
    )
    lift["overall_mean_value_score"] = value_df["value_score"].mean()
    lift["overall_mean_performance_to_price_ratio"] = value_df["performance_to_price_ratio"].mean()
    lift["lift_vs_overall"] = lift["recommendation_mean_value_score"] / lift["overall_mean_value_score"]
    lift["ppr_lift_vs_overall"] = (
        lift["recommendation_mean_performance_to_price_ratio"]
        / lift["overall_mean_performance_to_price_ratio"]
    )
    lift.to_csv(OUTPUT_DIR / "task_d_recommendation_lift.csv", index=False)

    segment_outliers = value_df.groupby("segment")["is_value_outlier"].agg(["sum", "count"]).reset_index()
    segment_outliers["outlier_ratio"] = segment_outliers["sum"] / segment_outliers["count"]
    segment_outliers.to_csv(OUTPUT_DIR / "task_d_outlier_segment_distribution.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=lift, x="scenario", y="ppr_lift_vs_overall")
    plt.axhline(1, color="black", linewidth=1)
    plt.title("Recommendation PPR lift")
    plt.xlabel("Scenario")
    plt.ylabel("PPR lift vs overall")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "recommendation_lift.png", dpi=160)
    plt.close()

    value_df.to_csv(OUTPUT_DIR / "df_with_segments.csv", index=False)
    return value_df, recommendations


def recommend_phones(
    df: pd.DataFrame,
    budget_eur: float,
    top_n: int = 10,
    segment_preference: str | None = None,
    brand_preference: str | None = None,
) -> pd.DataFrame:
    """Recommend Top-N phones by Performance-to-Price Ratio under a EUR budget constraint."""
    candidates = df[df["price_eur"] <= float(budget_eur)].copy()
    if segment_preference:
        candidates = candidates[candidates["segment"].eq(segment_preference)].copy()
    if brand_preference:
        candidates = candidates[candidates["brand_group"].eq(brand_preference)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[
            "recommendation_rank", "oem", "model", "brand_group", "price_eur", "budget_utilization",
            "price_tier", "segment", "spec_score_0_100", "value_score",
            "performance_to_price_ratio", "market_price_discount_ratio",
            "expected_market_price_eur", "market_underpriced_outlier",
            "is_value_outlier", "ram_gb",
            "storage_gb", "battery_capacity_mah", "main_camera_max_mp",
            "network_generation", "recommendation_reason",
        ])
    candidates["performance_to_price_ratio"] = candidates["spec_score_0_100"] / candidates["price_eur"]
    candidates["budget_utilization"] = candidates["price_eur"] / float(budget_eur)
    candidates["recommendation_reason"] = candidates.apply(make_recommendation_reason, axis=1)
    cols = [
        "recommendation_rank", "oem", "model", "brand_group", "price_eur", "budget_utilization",
        "price_tier", "segment", "spec_score_0_100", "value_score",
        "performance_to_price_ratio", "market_price_discount_ratio",
        "expected_market_price_eur", "market_underpriced_outlier",
        "is_value_outlier", "ram_gb",
        "storage_gb", "battery_capacity_mah", "main_camera_max_mp",
        "network_generation", "recommendation_reason",
    ]
    ranked = (
        candidates.sort_values(
            ["performance_to_price_ratio", "value_score", "is_value_outlier", "spec_score_0_100", "price_eur"],
            ascending=[False, False, False, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.insert(0, "recommendation_rank", range(1, len(ranked) + 1))
    return ranked[cols]


def make_recommendation_reason(row: pd.Series) -> str:
    """Build a short, presentation-friendly recommendation reason."""
    discount = row.get("market_price_discount_ratio", np.nan)
    discount_note = "" if pd.isna(discount) else f", market_discount={discount:.1%}"
    return (
        f"PPR={row['performance_to_price_ratio']:.3f}, value={row['value_score']:.2f}, "
        f"spec={row['spec_score_0_100']:.1f}{discount_note}, "
        f"RAM={row['ram_gb']:.0f}GB, camera={row.get('main_camera_max_mp', np.nan):.0f}MP"
    )


def save_price_driver_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact price-driver table for two-way solution demos."""
    summary = (
        importance_df.sort_values("importance", ascending=False)
        .head(20)
        .assign(
            driver_type=lambda d: np.where(
                d["feature"].str.startswith("cat__brand_group_"),
                "brand",
                "hardware_or_spec",
            )
        )
    )
    summary.to_csv(OUTPUT_DIR / "two_way_price_driver_summary.csv", index=False)
    return summary


def create_model_reference(price_model_df: pd.DataFrame, price_raw_df: pd.DataFrame) -> None:
    """Store simple raw mean/std references for CLI business price predictions."""
    numeric_cols = price_raw_df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        scaled_col = "num__" + col
        if scaled_col in price_model_df.columns:
            mean = float(price_raw_df[col].mean())
            std = float(price_raw_df[col].std(ddof=0))
            rows.append(
                {
                    "raw_feature": col,
                    "model_feature": scaled_col,
                    "mean": mean,
                    "std": std if std else 1.0,
                    "median": float(price_raw_df[col].median()),
                }
            )
    ref = pd.DataFrame(rows)
    ref.to_csv(MODEL_DIR / "numeric_feature_reference.csv", index=False)

    brand_groups = sorted(price_raw_df["brand_group"].dropna().unique().tolist())
    (MODEL_DIR / "brand_groups.json").write_text(
        json.dumps(brand_groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_business_demo() -> pd.DataFrame:
    """Create static business guide examples from trained artifacts."""
    from two_way_solution import TwoWaySmartphoneSolution

    solution = TwoWaySmartphoneSolution(root_dir=ROOT_DIR)
    rows = []
    examples = [
        {
            "scenario": "B2B_midrange_launch",
            "brand": "Samsung",
            "ram_gb": 8,
            "storage_gb": 128,
            "spec_score_0_100": 55,
            "battery_capacity_mah": 5000,
            "display_size_in": 6.5,
            "screen_to_body_pct": 84,
            "resolution_width_px": 1080,
            "resolution_height_px": 2400,
            "ppi": 405,
            "body_weight_g": 190,
            "fast_charging_w": 25,
            "main_camera_max_mp": 50,
            "network_generation": 5,
            "sensor_count": 5,
            "launch_year": 2020,
            "budget_price_eur": 450,
        },
        {
            "scenario": "B2B_flagship_launch",
            "brand": "Apple",
            "ram_gb": 8,
            "storage_gb": 256,
            "spec_score_0_100": 72,
            "battery_capacity_mah": 4300,
            "display_size_in": 6.7,
            "screen_to_body_pct": 87,
            "resolution_width_px": 1284,
            "resolution_height_px": 2778,
            "ppi": 460,
            "body_weight_g": 228,
            "fast_charging_w": 27,
            "main_camera_max_mp": 48,
            "network_generation": 5,
            "sensor_count": 6,
            "launch_year": 2020,
            "budget_price_eur": 950,
        },
        {
            "scenario": "B2B_value_launch",
            "brand": "Xiaomi",
            "ram_gb": 6,
            "storage_gb": 128,
            "spec_score_0_100": 50,
            "battery_capacity_mah": 5000,
            "display_size_in": 6.4,
            "screen_to_body_pct": 83,
            "resolution_width_px": 1080,
            "resolution_height_px": 2400,
            "ppi": 395,
            "body_weight_g": 195,
            "fast_charging_w": 30,
            "main_camera_max_mp": 64,
            "network_generation": 5,
            "sensor_count": 5,
            "launch_year": 2020,
            "budget_price_eur": 300,
        },
    ]
    for spec in examples:
        result = solution.business_price_guide(spec)
        rows.append({**spec, **result})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "two_way_business_price_guides.csv", index=False)
    return df


def write_docs(
    data_source: Path,
    regression: RegressionResult,
    importance_df: pd.DataFrame,
    adjusted_brand: pd.DataFrame,
    segmented: pd.DataFrame,
    value_df: pd.DataFrame,
) -> None:
    """Write README and audit docs after all outputs are available."""
    def df_to_md(df: pd.DataFrame, max_rows: int = 12) -> str:
        """Render a compact markdown table without optional tabulate dependency."""
        compact = df.head(max_rows).copy()
        headers = [str(c) for c in compact.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in compact.iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    best = regression.best_row
    top_features = (
        importance_df.sort_values("importance", ascending=False)["feature"].head(5).tolist()
    )
    segment_summary = pd.read_csv(OUTPUT_DIR / "cluster_segment_summary.csv")
    outlier_validation = pd.read_csv(OUTPUT_DIR / "value_outlier_validation_summary.csv")
    outlier_methods = pd.read_csv(OUTPUT_DIR / "value_outlier_method_comparison.csv")
    rec_validation = pd.read_csv(OUTPUT_DIR / "recommendation_constraint_validation.csv")
    recommendations = pd.read_csv(OUTPUT_DIR / "recommendations_all_scenarios.csv")
    business_guides = pd.read_csv(OUTPUT_DIR / "two_way_business_price_guides.csv")
    cv_fold = pd.read_csv(OUTPUT_DIR / "regression_cv_fold_metrics.csv")
    data_source_label = "content/"
    outlier_false = outlier_validation[outlier_validation["is_value_outlier"].eq(False)].iloc[0]
    outlier_true = outlier_validation[outlier_validation["is_value_outlier"].eq(True)].iloc[0]

    quality_checks = pd.DataFrame(
        [
            {
                "check": "uses_github_root_content_folder",
                "status": "PASS" if data_source.exists() else "FAIL",
                "evidence": str(data_source),
            },
            {
                "check": "regression_model_beats_median_baseline",
                "status": "PASS" if float(best["MASE"]) < 1 else "WARN",
                "evidence": f"best={best['model']} {best['target_mode']}, MAE={best['MAE']:.2f}, MASE={best['MASE']:.3f}",
            },
            {
                "check": "five_fold_cv_available",
                "status": "PASS" if cv_fold["fold"].nunique() == 5 else "FAIL",
                "evidence": f"folds={cv_fold['fold'].nunique()}, rows={len(cv_fold)}",
            },
            {
                "check": "three_market_segments_created",
                "status": "PASS" if set(segment_summary["segment"]) == {"Entry", "Mid-range", "Flagship"} else "FAIL",
                "evidence": segment_summary[["segment", "count"]].to_dict("records"),
            },
            {
                "check": "value_outliers_are_cheaper_than_non_outliers",
                "status": "PASS" if outlier_true["avg_price_eur"] < outlier_false["avg_price_eur"] else "FAIL",
                "evidence": f"outlier={outlier_true['avg_price_eur']:.2f}, non_outlier={outlier_false['avg_price_eur']:.2f}",
            },
            {
                "check": "value_outliers_have_higher_ppr",
                "status": "PASS" if outlier_true["avg_performance_to_price_ratio"] > outlier_false["avg_performance_to_price_ratio"] else "FAIL",
                "evidence": f"outlier={outlier_true['avg_performance_to_price_ratio']:.3f}, non_outlier={outlier_false['avg_performance_to_price_ratio']:.3f}",
            },
            {
                "check": "value_outliers_are_under_expected_market_price",
                "status": "PASS" if outlier_true["avg_market_price_gap_eur"] > 0 else "FAIL",
                "evidence": f"avg_gap={outlier_true['avg_market_price_gap_eur']:.2f}, avg_discount={outlier_true['avg_market_price_discount_ratio']:.1%}",
            },
            {
                "check": "recommendations_respect_budget",
                "status": "PASS" if bool(rec_validation["all_within_budget"].all()) else "FAIL",
                "evidence": rec_validation[["scenario", "budget_eur", "max_price_eur", "all_within_budget"]].to_dict("records"),
            },
            {
                "check": "recommendations_sorted_by_ppr",
                "status": "PASS" if bool(rec_validation["is_performance_to_price_ratio_descending"].all()) else "FAIL",
                "evidence": rec_validation[["scenario", "is_performance_to_price_ratio_descending", "top_performance_to_price_ratio"]].to_dict("records"),
            },
            {
                "check": "business_price_guides_created",
                "status": "PASS" if len(business_guides) >= 3 else "FAIL",
                "evidence": business_guides[["scenario", "brand", "predicted_price_eur", "guidance"]].to_dict("records"),
            },
        ]
    )
    quality_checks.to_csv(OUTPUT_DIR / "modeling_quality_checks.csv", index=False)

    readme = f"""# DS Team 7 GSM 모델링 요약

## 사용 데이터
- 입력 위치: `{data_source_label}`
- 회귀 학습: `content/gsm_processed_all(price_prediction).csv`
- 추천/클러스터링: `content/gsm_processed(recommendation).csv`
- 가격 단위: EUR
- 회귀 타겟: `target_price_eur`

## Task A. 가격 예측 회귀
최종 holdout 기준 최고 모델은 **{best['model']} ({best['target_mode']})** 이다.
MAE는 **{best['MAE']:.2f} EUR**, R2는 **{best['R2']:.4f}** 이다.

상위 가격 결정 요인:
{', '.join(top_features)}

## Task B. 세그먼트 클러스터링
KMeans k=3으로 Entry / Mid-range / Flagship을 만들었다.

{df_to_md(segment_summary)}

## Task C. 가성비 이상치와 추천
가성비 우수 모델은 세 가지 기준을 결합해 식별한다.
1. 같은 세그먼트 내 `value_score` Z-score 이상치
2. `IsolationForest` 기반 이상치 중 value score 상위 후보
3. 하드웨어 스펙으로 예측한 `expected_market_price_eur`보다 실제 가격이 낮은 market underpricing 이상치

추천은 사용자 예산 이하 후보만 남긴 뒤 `performance_to_price_ratio`를 첫 번째 정렬 기준으로 최적 대안을 고른다.

{df_to_md(outlier_validation)}

추천 검증:

{df_to_md(rec_validation)}

## 직접 실행
```bash
cd modeling
python3 two_way_solution.py --mode demo
python3 two_way_solution.py --mode recommend --budget-eur 400 --top-n 5
```

## 주요 산출물
- `models/best_price_regressor.pkl`
- `outputs/regression_holdout_metrics.csv`
- `outputs/regression_cv_metrics.csv`
- `outputs/regression_cv_fold_metrics.csv`
- `outputs/feature_importance_best_target.csv`
- `outputs/brand_premium_summary.csv`
- `outputs/adjusted_brand_premium_summary.csv`
- `outputs/df_with_segments.csv`
- `outputs/value_outliers.csv`
- `outputs/value_outlier_method_comparison.csv`
- `outputs/recommendations_all_scenarios.csv`
- `outputs/modeling_quality_checks.csv`
- `outputs/two_way_business_price_guides.csv`
- `outputs/two_way_user_recommendations.csv`
- `docs/02_MODEL_EVALUATION_HANDOFF.md`

## 한계
- 가격은 EUR 기준이며 지역/통신사/유통 채널 차이는 반영하지 않는다.
- 일부 스펙은 텍스트에서 파싱된 값이므로 원본 표기 품질의 영향을 받는다.
- GSM 최종 전처리 파일은 CPU 벤치마크를 직접 제공하지 않아 RAM, storage, camera, battery, network, display 계열 피처와 `spec_score_0_100`을 하드웨어 성능 대리 지표로 사용한다.
- 구형 모델과 최신 모델이 함께 있어 `phone_age` 해석이 필요하다.
"""
    (MODELING_DIR / "README_modeling.md").write_text(readme, encoding="utf-8")

    usage_audit = """# 00. 팀원 전처리 파일 사용 감사

## 확인 결과
- 모델링은 raw GSM CSV를 다시 전처리하지 않고 팀원이 만든 `content/` 전처리 CSV를 입력으로 사용한다.
- 전처리 설명 문서 `Preprocessing/설명.docx`와 전처리 노트북 `Preprocessing/GSM_전처리+피처엔지니어링.ipynb`를 확인했다.
- 회귀 모델은 인코딩/스케일링 완료 파일 `content/gsm_processed_all(price_prediction).csv`를 사용한다.
- 추천과 발표용 출력은 원 단위 파일 `content/gsm_processed(recommendation).csv`를 사용한다.
- ZIP 파일을 입력 데이터로 사용하지 않고 GitHub 루트의 `content/` 경로를 직접 사용한다.

## 모델링 입력 연결
| 작업 | 사용 파일 |
| --- | --- |
| Regression | `content/gsm_processed_all(price_prediction).csv` |
| Brand premium | `content/gsm_processed(price_prediction).csv` |
| Clustering | `content/gsm_processed(recommendation).csv` |
| Outlier/Recommendation | `content/gsm_processed(recommendation).csv` |
"""
    (DOC_DIR / "00_TEAM_FILE_USAGE_AUDIT.md").write_text(usage_audit, encoding="utf-8")

    dataset_summary = f"""# 01. 데이터셋과 전처리 요약

## 새 데이터셋
- 원본 GSM 데이터: 10,679 rows x 86 columns
- 중복 제거 후: 10,105 rows
- 가격 예측 가능 행: 6,251 rows
- 추천 가능 행: 2,113 rows

## Dirty data 처리
- `-`, `N/A`, `null`, `None`, 빈 문자열을 NaN으로 통일
- 깨진 특수문자 복원
- 스펙 문자열에서 수치형 피처 파싱
- 비정상 범위 값은 NaN 처리 후 imputation
- 추천 품질을 위해 `price_eur >= 50`, `ram_gb >= 2` 필터 적용

## 인코딩/스케일링
- One-Hot Encoding: `brand_group`, `launch_status_group`, `display_panel`, `battery_type`, `os_family`
- StandardScaler: numeric features

## 과제 조건 대응
수치형과 범주형 데이터가 모두 있고, 결측/문자열/이상치/깨진 특수문자 등 dirty data 처리 과정이 존재한다.
"""
    (DOC_DIR / "01_DATASET_AND_PREPROCESSING_SUMMARY.md").write_text(dataset_summary, encoding="utf-8")

    handoff = f"""# 02. 모델 평가 담당자 인수인계

## 평가 대상
모델링 담당자가 만든 GSM 기반 가격 예측/추천 모델이다.

## 회귀 성능
최고 모델: `{best['model']} ({best['target_mode']})`

| MAE | RMSE | sMAPE | MASE | R2 |
| ---: | ---: | ---: | ---: | ---: |
| {best['MAE']:.2f} | {best['RMSE']:.2f} | {best['sMAPE']:.2f} | {best['MASE']:.3f} | {best['R2']:.4f} |

확인 파일:
- `outputs/regression_holdout_metrics.csv`
- `outputs/regression_cv_metrics.csv`
- `outputs/regression_cv_fold_metrics.csv`
- `models/best_price_regressor.pkl`

## 해석 확인
- 가격 결정 요인: `outputs/feature_importance_best_target.csv`
- 브랜드 중요도: `outputs/brand_importance_summary.csv`
- 동일 사양 대비 브랜드 프리미엄: `outputs/adjusted_brand_premium_summary.csv`

## 추천 확인
- 가성비 이상치: `outputs/value_outlier_validation_summary.csv`
- 예산 제약 검증: `outputs/recommendation_constraint_validation.csv`
- 추천 결과: `outputs/recommendations_all_scenarios.csv`

## 평가자가 볼 체크포인트
1. 회귀 MAE/R2가 baseline보다 충분히 좋은가?
2. CV와 holdout 성능 차이가 과도하지 않은가?
3. Feature importance가 하드웨어/브랜드 요인으로 해석 가능한가?
4. 세그먼트가 Entry/Mid-range/Flagship으로 시장 해석이 가능한가?
5. 가성비 이상치 그룹이 실제로 평균 가격은 낮고 value score/PPR은 높은가?
6. 가성비 이상치 그룹이 expected market price 대비 할인되어 있는가?
7. 추천 결과가 예산 이하이고 Performance-to-Price Ratio 내림차순인가?
"""
    (DOC_DIR / "02_MODEL_EVALUATION_HANDOFF.md").write_text(handoff, encoding="utf-8")

    evaluation_guide = f"""# 06. 모델 평가 담당자용 상세 가이드

## 목적
이 문서는 모델링 담당자가 만든 GSM 스마트폰 가격 예측/추천 모델을 평가 담당자가 빠르게 검증할 수 있도록 정리한 인수인계 문서다.

## 사용 데이터와 경로
- 기준 데이터 위치: `{data_source_label}`
- 회귀 입력: `content/gsm_processed_all(price_prediction).csv`
- 브랜드 프리미엄 해석 입력: `content/gsm_processed(price_prediction).csv`
- 클러스터링/이상치/추천 입력: `content/gsm_processed(recommendation).csv`
- 원본/EDA 참고: `content/gsm.csv`, `inspection_data/`

## 전체 모델 흐름
1. 팀원 전처리 CSV를 읽는다.
2. 가격 예측 회귀 모델을 학습한다.
3. 회귀 feature importance와 동일 사양 대비 브랜드 프리미엄을 산출한다.
4. 추천용 데이터에서 KMeans로 Entry / Mid-range / Flagship 세그먼트를 만든다.
5. PPR, value score, expected market price 기반으로 가성비 우수 이상치를 탐지한다.
6. 사용자 예산 이하 후보를 남기고 `performance_to_price_ratio` 내림차순으로 추천한다.
7. 기업용 `business_price_guide()`와 소비자용 `recommend_for_user()`로 양방향 솔루션을 실행한다.

## 최종 가격 예측 모델
- 최고 모델: `{best['model']} ({best['target_mode']})`
- Holdout MAE: `{best['MAE']:.2f} EUR`
- Holdout RMSE: `{best['RMSE']:.2f} EUR`
- Holdout sMAPE: `{best['sMAPE']:.2f}`
- Holdout MASE: `{best['MASE']:.3f}`
- Holdout R2: `{best['R2']:.4f}`
- 확인 파일: `outputs/regression_holdout_metrics.csv`, `outputs/regression_cv_metrics.csv`, `outputs/regression_cv_fold_metrics.csv`

## 가격 결정 요인과 브랜드 인지도
상위 가격 결정 요인은 다음과 같다.

{df_to_md(pd.DataFrame({"top_feature": top_features}), max_rows=10)}

브랜드 관련 확인 파일:
- `outputs/brand_importance_summary.csv`
- `outputs/brand_premium_summary.csv`
- `outputs/adjusted_brand_premium_summary.csv`

평가 시 볼 점:
- 하드웨어 피처가 feature importance 상위권에 나타나는가?
- 브랜드 그룹이 가격 차이에 어느 정도 영향을 주는가?
- adjusted premium은 동일 사양 대비 잔차 프리미엄으로 해석 가능한가?

## 가성비 이상치 탐지
가성비 우수 모델은 다음 세 기준의 union으로 잡았다.

{df_to_md(outlier_methods)}

핵심 검증 결과:

{df_to_md(outlier_validation)}

해석:
- 이상치 그룹 평균 가격은 `{outlier_true['avg_price_eur']:.2f} EUR`로 비이상치 그룹 `{outlier_false['avg_price_eur']:.2f} EUR`보다 낮다.
- 이상치 그룹 평균 PPR은 `{outlier_true['avg_performance_to_price_ratio']:.3f}`로 비이상치 그룹 `{outlier_false['avg_performance_to_price_ratio']:.3f}`보다 높다.
- 이상치 그룹은 expected market price 대비 평균 `{outlier_true['avg_market_price_discount_ratio']:.1%}` 저평가되어 있다.

평가 시 볼 파일:
- `outputs/value_outliers.csv`
- `outputs/value_outlier_method_comparison.csv`
- `outputs/value_outlier_validation_summary.csv`
- `outputs/df_with_segments.csv`

## 예산 내 추천 알고리즘
추천 알고리즘은 `price_eur <= budget_eur`로 후보를 제한한 뒤 다음 순서로 정렬한다.

1. `performance_to_price_ratio` 내림차순
2. `value_score` 내림차순
3. `is_value_outlier` 우선
4. `spec_score_0_100` 내림차순
5. `price_eur` 오름차순

추천 검증:

{df_to_md(rec_validation)}

추천 예시 상위 결과:

{df_to_md(recommendations.head(10))}

평가 시 볼 점:
- 모든 추천 가격이 예산 이하인가?
- `is_performance_to_price_ratio_descending`가 True인가?
- 추천 사유에 PPR과 market discount가 표시되는가?
- 사용자가 예산을 직접 입력할 수 있는가?

## 직접 실행 명령
```bash
cd DS_Team7/modeling
python3 run_modeling.py --run
python3 two_way_solution.py --mode demo
python3 two_way_solution.py --mode recommend --budget-eur 400 --top-n 5
python3 two_way_solution.py --mode recommend --top-n 5
```

마지막 명령은 `--budget-eur`를 생략했기 때문에 실행 중 예산을 직접 입력한다.

## 모델 평가 체크리스트
{df_to_md(quality_checks)}

## 남은 한계와 발표 시 주의점
- 가격 단위는 EUR이며 지역/통신사/유통 채널 차이는 반영하지 않았다.
- CPU 벤치마크가 최종 전처리 파일에 직접 수치화되어 있지 않아 RAM, storage, camera, battery, network, display 계열 피처와 `spec_score_0_100`을 하드웨어 성능 대리 지표로 사용했다.
- GSM 데이터는 출시연도와 구형 모델이 섞여 있어 `phone_age` 해석이 필요하다.
- 최종 팀 제출물에는 모델링 외에도 PPT, 팀원별 contribution, 팀원별 learned writeup, 출처 표기가 별도로 필요하다.
"""
    (DOC_DIR / "06_MODEL_EVALUATION_GUIDE_FOR_TEAM.md").write_text(evaluation_guide, encoding="utf-8")

    library_doc = """# 04. 외부 라이브러리와 모델 설명

과제 설명 PDF의 요구사항에 맞춰, 수업 외 라이브러리와 주요 모델/메서드의 역할을 정리한다.

## pandas / numpy
- 사용 위치: 모든 CSV 로딩, 결측치 집계, 그룹별 평균/중앙값, 추천 정렬.
- 주요 출력: `outputs/*.csv`.

## scikit-learn
- `train_test_split(test_size=0.2, random_state=42)`: 회귀 holdout 평가용 train/test 분리. 전체 가격 예측 데이터를 80:20으로 나눠 최종 모델의 일반화 오차를 확인한다.
- `KFold(n_splits=5, shuffle=True, random_state=42)`: 회귀 모델 안정성 검증용 5-fold CV. 과제 PDF는 classification에 k-fold를 요구하지만, 본 프로젝트는 classification을 쓰지 않으므로 regression에도 추가 검증으로 적용했다.
- `LinearRegression()`: 선형 회귀 baseline. 하드웨어 스펙과 가격 사이의 단순 선형 관계를 비교 기준으로 사용한다.
- `Ridge(alpha=1.0, random_state=42)`: L2 정규화 선형 회귀. 다수의 one-hot/스케일링 피처에서 계수 과대화를 줄이는 baseline이다.
- `RandomForestRegressor(n_estimators=220, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)`: 비선형 가격 예측 후보 모델. 여러 decision tree를 bagging해 스펙 간 상호작용을 포착한다.
- `RandomForestRegressor(n_estimators=260, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)`: feature importance 산출용 보조 모델.
- `RandomForestRegressor(n_estimators=140, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)`: expected market price 산출용 보조 모델. out-of-fold 방식으로 각 모델의 시장 기대가를 추정한다.
- `KMeans(n_clusters=3, n_init=10, random_state=42)`: Entry/Mid-range/Flagship 3개 세그먼트 분류. `n_init=10`은 초기 중심점 선택을 10번 반복해 안정적인 군집을 고르는 설정이다.
- `AgglomerativeClustering(n_clusters=3, linkage="ward")`: KMeans 세그먼트 품질 비교용 보조 클러스터링. ward linkage는 군집 내 분산 증가를 최소화한다.
- `IsolationForest(contamination=0.05, random_state=42)`: 시장 평균 대비 비정상적으로 높은 value score/PPR과 underpricing을 보이는 가성비 후보 탐지. `contamination=0.05`는 약 5%를 이상치 후보로 보는 설정이다.
- `SimpleImputer(strategy="median")`: 클러스터링/시장 기대가/이상치 입력값의 결측 수치형 값을 중앙값으로 대체한다.
- `StandardScaler()`: 클러스터링과 이상치 탐지 입력값을 평균 0, 표준편차 1로 변환해 단위 차이가 거리 계산을 지배하지 않게 한다.
- `PCA(n_components=2, random_state=42)`: 클러스터링 결과 시각화를 위한 2차원 축소.
- `mean_absolute_error`, `mean_squared_error`, `r2_score`, `silhouette_score`, `davies_bouldin_score`: 회귀 모델과 클러스터링 품질 평가에 사용한다.

## LightGBM
- 사용 모델: `LGBMRegressor`.
- 목적: tabular smartphone spec 데이터에서 비선형 가격 패턴을 학습하는 최종 후보 모델.
- 주요 파라미터: `n_estimators=450`, `learning_rate=0.045`, `num_leaves=31`, `subsample=0.9`, `colsample_bytree=0.9`, `random_state=42`, `n_jobs=-1`, `verbose=-1`.
- `n_estimators`는 boosting tree 수, `learning_rate`는 각 tree의 반영 비율, `num_leaves`는 tree 복잡도, `subsample`과 `colsample_bytree`는 행/열 샘플링 비율이다.
- 실행 환경에 LightGBM이 없으면 `HistGradientBoostingRegressor(learning_rate=0.045, max_iter=450, max_leaf_nodes=31, random_state=42)`로 자동 대체한다.

## joblib
- 사용 목적: 최종 가격 예측 모델을 `models/best_price_regressor.pkl`로 저장하고 `two_way_solution.py`에서 재사용.

## matplotlib / seaborn
- 사용 목적: feature importance, 브랜드 프리미엄, 클러스터 품질, PCA 세그먼트, 가성비 이상치, 추천 lift 시각화.
- 주요 출력: `outputs/plots/*.png`.
"""
    (DOC_DIR / "04_EXTERNAL_LIBRARY_METHOD_EXPLANATIONS.md").write_text(library_doc, encoding="utf-8")

    compliance_rows = [
        ["Spec", "team project", "Team 7 repository and PDF team list", "PASS"],
        ["Spec", "proposal submitted", "Term_Project_Proposal.pdf / DS_Proposal_한글화.pdf", "PASS"],
        ["Spec", "proposal includes dataset/objective/algorithm types", "proposal docs + modeling docs + dataset-change alignment note", "PASS_WITH_NOTE"],
        ["Spec", "end-to-end Big Data process", "Preprocessing, inspection, modeling, evaluation outputs", "PASS"],
        ["Spec", ">=10 records and features", "raw GSM 10,679 x 86; processed data >2,000 rows and >50 columns", "PASS"],
        ["Spec", "dirty data", "missing values, mixed-unit text specs, price strings, duplicate/missing inspection", "PASS"],
        ["Spec", "numerical+categorical", "numeric hardware/price plus brand/status/panel/battery/os categorical features", "PASS"],
        ["Spec", "scaling and encoding", "OneHotEncoder + StandardScaler in preprocessing; StandardScaler in modeling", "PASS"],
        ["Spec", "2 of regression/classification/clustering", "Regression + Clustering", "PASS"],
        ["Spec", "k-fold CV for classification", "no classification model; regression 5-fold CV added as extra validation", "NOT_APPLICABLE_WITH_EXTRA"],
        ["Spec", "PPT presentation", "presentation/DS_Team7_GSM_Final_Presentation.pptx + outline doc", "PASS"],
        ["Spec", "separate writeup behind PPT", "handoff, evaluation guide, compliance audit docs", "PASS"],
        ["Spec", "source code with detailed comments", "run_modeling.py, two_way_solution.py, GSM modeling notebook", "PASS"],
        ["Spec", "external library/method explanations", "04_EXTERNAL_LIBRARY_METHOD_EXPLANATIONS.md", "PASS"],
        ["Spec", "outputs including plots and execution results", "outputs CSV/PNG/model artifacts", "PASS"],
        ["Spec", "dataset used", "content/*.csv included", "PASS"],
        ["Spec", "teamwork task assignment and contribution percentage", "template added; actual percentages required", "TEAM_INPUT_REQUIRED"],
        ["Spec", "learning writeup for each member", "template added; each member must fill", "TEAM_INPUT_REQUIRED"],
        ["Spec", "cite Internet/blog/Kaggle/GitHub sources", "Kaggle GSMArena Mobile Phone Devices source and library docs listed", "PASS_WITH_SOURCE_CHECK"],
        ["Proposal", "기업 가격 가이드", "best_price_regressor + two_way_solution.py", "PASS"],
        ["Proposal", "소비자 추천", "budget constraint + PPR rank optimization", "PASS"],
        ["Proposal", "브랜드 프리미엄", "importance + adjusted residual premium", "PASS"],
        ["Proposal", "가성비 이상치", "Z-score + IsolationForest + market underpricing", "PASS"],
        ["Proposal", "데이터셋 변경 정합성", "12_DATASET_CHANGE_AND_PROPOSAL_ALIGNMENT.md explains why GSM replaces the proposal-time synthetic dataset", "PASS_WITH_NOTE"],
    ]
    compliance = pd.DataFrame(compliance_rows, columns=["source", "requirement", "evidence", "status"])
    compliance.to_csv(OUTPUT_DIR / "proposal_spec_compliance_matrix.csv", index=False)

    data_change_doc = """# 12. Dataset Change and Proposal Alignment

## Why This File Exists

The proposal PDF originally listed the Kaggle `Smartphones-specification-and-prices` dataset. The final project now uses the team-preprocessed GSM/GSMArena smartphone dataset in `content/`. This is not a topic change. It is a dataset replacement inside the same project objective: smartphone price guidance for companies and budget-based device recommendation for consumers.

## Reason for the Final GSM Dataset

The term-project specification requires a dataset with at least 10 records/features, dirty data, and a mixture of numerical and categorical attributes. The final GSM dataset better supports those grading conditions because it contains:

- raw data with 10,679 rows and 86 specification columns;
- many real-world dirty fields such as missing values, mixed-unit hardware strings, price text, duplicated model names, and sparse specification columns;
- categorical variables such as OEM/brand, OS, display panel, launch status, battery type, network support, and sensor-related fields;
- numerical variables and engineered hardware indicators such as price in EUR, RAM, storage, camera MP, battery capacity, display size/resolution, body weight/thickness, launch year, and spec/value scores.

## Proposal Goal Mapping

| Proposal requirement | Final GSM implementation |
| --- | --- |
| Analyze hardware indicators and brand recognition impact on price | Regression model, feature importance, adjusted brand premium, and `two_way_price_driver_summary.csv` |
| Provide objective company price guide | `best_price_regressor.pkl` and `two_way_solution.py --mode price-guide` |
| Recommend optimal models within a user's budget | `two_way_solution.py --mode recommend --budget-eur ...` and scenario outputs |
| Regression and feature importance | Linear/Ridge/RandomForest/LightGBM comparison, holdout metrics, 5-fold CV, feature importance plots |
| Brand premium proof | Brand one-hot features plus residual-based adjusted brand premium summary |
| Entry/Mid-range/Flagship segmentation | KMeans k=3 segment assignment and segment summary |
| Value-for-money outlier detection | Segment Z-score, IsolationForest, and expected-market-price underpricing |
| Rank optimization under budget constraint | Filter `price_eur <= budget`, then sort by PPR/value score/outlier/spec score/lower price |

## Important Interpretation Note

The proposal mentions CPU and processor performance. The final GSM dataset has chipset/CPU text fields in the raw data, but the cleaned modeling table does not contain a reliable universal CPU benchmark. Therefore the model uses available hardware proxies and engineered performance indicators: RAM, storage, camera, battery, display resolution/size, network support, sensor count, and aggregate `spec_score_0_100`. This should be stated in presentation/Q&A as a limitation and as a practical decision caused by the real dataset schema.

## Submission Position

The final data change is acceptable for the term-project specification because the specification allows selecting a suitable dataset independently, and the final dataset more directly demonstrates the required dirty-data inspection, encoding, scaling, regression, clustering, evaluation, plots, and practical two-way solution outputs.
"""
    (DOC_DIR / "12_DATASET_CHANGE_AND_PROPOSAL_ALIGNMENT.md").write_text(data_change_doc, encoding="utf-8")

    risk = f"""# 03. 최종 루브릭 리스크 감사

## 최종 판단
현재 GSM 모델링/분석 산출물 기준으로 과제 평가에 치명적인 기술 조건 누락은 발견되지 않는다. PPT 제출본, 데이터 출처 체크리스트, 데이터셋 변경 정합성 문서까지 보강했다. 다만 PDF의 최종 제출 패키지 조건 중 팀원별 기여도와 각 팀원 학습회고는 팀원이 사실 정보를 확인해 채워야 하는 항목이다.

## PASS 근거
- Regression + Clustering 사용
- Scaling/Encoding 사용 증거 존재
- 회귀 모델 4종과 로그 타겟 비교
- 5-fold CV 수행
- Feature importance와 브랜드 프리미엄 산출
- 가성비 이상치와 예산 추천 직접 구현
- expected market price 대비 저평가 여부와 PPR 최대화 정렬 검증
- 코드 실행 결과, CSV, PNG, pkl, 문서 산출물 생성
- fold별 CV 결과와 외부 라이브러리 설명 문서 추가
- 제안서 당시 synthetic 데이터셋과 최종 GSM 데이터셋 차이를 별도 문서로 설명

## 남은 팀 단위 필요 사항
- 최종 PPT: `presentation/DS_Team7_GSM_Final_Presentation.pptx` 생성 완료. 발표 담당자가 디자인/팀 정보만 최종 확인
- 팀원별 contribution percentage: `modeling/docs/08_TEAMWORK_CONTRIBUTION_AND_LEARNING_TEMPLATE.md`에 실제 값 입력
- 팀원별 learned writeup: `modeling/docs/08_TEAMWORK_CONTRIBUTION_AND_LEARNING_TEMPLATE.md`에 각자 작성
- 인터넷 코드/자료 출처 최종 표기: `modeling/docs/09_SOURCE_CITATION_CHECKLIST.md`에 Kaggle GSMArena Mobile Phone Devices와 공식 라이브러리 문서 출처 반영

## 모델링 한계
- 가격은 EUR 기준
- 일부 스펙은 텍스트 파싱값
- 지역/통신사/유통 채널 가격 차이 미반영
- Proposal PDF의 과거 synthetic 데이터 설명과 실제 최종 GSM 데이터셋이 다르므로 발표에서 `12_DATASET_CHANGE_AND_PROPOSAL_ALIGNMENT.md`의 변경 사유를 명확히 설명해야 한다.
"""
    (DOC_DIR / "03_FINAL_RUBRIC_RISK_AUDIT.md").write_text(risk, encoding="utf-8")

    term_spec_audit = """# 07. Term Project Specification Compliance Audit

기준 문서: `TermProject_specification_c1_0507.pdf`

## 최종 판단

모델링/분석 산출물은 과제 설명 PDF의 핵심 기술 조건을 충족한다. PPT 제출본과 데이터 출처 표기도 보강했다. 다만 최종 제출 패키지 관점에서는 팀원별 contribution percentage와 각 팀원 learning writeup이 팀 차원에서 반드시 채워져야 한다. 이 두 가지는 모델링 코드로 임의 생성하면 안 되는 사실 정보라서 별도 템플릿을 추가했다.

## 세부 조건 대조표

{compliance_table}

## 평가상 주의할 점

1. 최종 데이터셋이 proposal 당시 설명과 다르므로, 발표에서 "과제 조건의 dirty data, numeric/categorical, 충분한 records/features 조건을 더 잘 만족하기 위해 GSM 데이터로 변경했다"라고 명확히 말하고 `12_DATASET_CHANGE_AND_PROPOSAL_ALIGNMENT.md`를 근거로 둔다.
2. 모델링 성능은 회귀 R2만 단독으로 강조하기보다 MAE, MASE, CV, 가격가이드/추천 활용성을 함께 설명하는 편이 안전하다.
3. 팀워크/학습회고는 과제 설명 PDF에 직접 적힌 제출 항목이므로, 최종 제출 전 반드시 실제 값으로 채워야 한다.
""".format(compliance_table=df_to_md(compliance, max_rows=len(compliance)))
    (DOC_DIR / "07_TERM_PROJECT_SPEC_COMPLIANCE_AUDIT.md").write_text(term_spec_audit, encoding="utf-8")

    teamwork_template = """# 08. Teamwork Contribution and Learning Writeup Template

과제 설명 PDF의 "Teamwork data: task assignment and contribution percentage for each member"와 "A short writeup on what you have learned for each member" 조건을 채우기 위한 최종 제출용 템플릿이다. 아래 값은 사실 기반으로 팀원이 직접 확인해서 채워야 하며, 임의 작성하면 안 된다.

## Teamwork Contribution

| Member | Main task assignment | Detailed contribution | Contribution percentage |
| --- | --- | --- | --- |
| 홍지원 (202334357) | Inspection / Evaluation | 제안서 역할 기준. 원본 GSM inspection 자료, 평가/검증 결과 확인, 최종 발표 전 모델 평가 담당 범위 입력 필요 | 팀 입력 필요 |
| 강지윤 (202334414) | Preprocessing | 제안서 역할 기준. GSM 전처리/feature engineering 노트북과 설명 문서 담당 범위 입력 필요 | 팀 입력 필요 |
| 신준하 (202234905) | Analysis / Modeling | 제안서 역할 기준. GSM price regression, feature importance, brand premium analysis, clustering, value-outlier detection, budget recommendation, evaluation handoff | 팀 입력 필요 |
| Total |  |  | 100% |

주의: 과제 설명 PDF의 Team 7 명단에는 `홍지원` 이름이 두 번 표시되어 있으나, 팀 제안서에는 `202334357 홍지원`, `202334414 강지윤`, `202234905 신준하` 3명이 명시되어 있다. 최종 제출물은 제안서의 학번/역할 기준으로 작성하고, 실제 팀원이 추가로 있었다면 팀 확인 후 표를 보완한다.

## Individual Learning Writeups

### 홍지원 (202334357)

- Learned:
- Difficulty:
- Contribution reflection:

### 강지윤 (202334414)

- Learned:
- Difficulty:
- Contribution reflection:

### 신준하 (202234905)

- Learned: 가격 예측 회귀, feature importance 해석, 브랜드 프리미엄 분석, KMeans 세그먼트 분류, IsolationForest와 시장 기대가 기반 가성비 이상치 탐지, 예산 제약 추천 알고리즘을 GSM 스마트폰 데이터에 연결하는 방법을 학습했다.
- Difficulty: 전처리된 스펙 피처와 실제 가격 단위가 섞여 있어 모델 성능만이 아니라 해석 가능성과 추천 검증 기준을 함께 맞추는 점이 어려웠다.
- Contribution reflection: 기업용 적정 가격 가이드와 소비자용 예산 추천이 같은 데이터/모델 산출물에서 이어지도록 모델링 파이프라인과 평가 문서를 정리했다.

## Final Submission Checklist

- Contribution percentage 합계가 100%인지 확인한다.
- 각 팀원 learning writeup이 비어 있지 않은지 확인한다.
- 제안서의 3명 학번/역할과 PPT의 역할 분담 슬라이드가 충돌하지 않는지 확인한다.
- PPT의 역할 분담 슬라이드와 이 표의 내용이 충돌하지 않는지 확인한다.
"""
    (DOC_DIR / "08_TEAMWORK_CONTRIBUTION_AND_LEARNING_TEMPLATE.md").write_text(teamwork_template, encoding="utf-8")

    citation_checklist = """# 09. Source Citation Checklist

과제 설명 PDF는 인터넷 코드, Kaggle, GitHub, 블로그 등을 사용한 경우 반드시 출처를 표기하라고 요구한다. 아래 목록을 최종 보고서/PPT의 References에 반영한다.

## Must Confirm Before Submission

| Item | Current status | Required action |
| --- | --- | --- |
| Original GSM dataset download page | PASS_WITH_SOURCE_CHECK | Kaggle `GSMArena Mobile Phone Devices` by Mohit Sainani: https://www.kaggle.com/datasets/msainani/gsmarena-mobile-devices. The dataset description matches the project raw data shape: 10,000+ phone models, 86 specification fields, GSMArena extraction. |
| Preprocessing reference code | PASS_NO_EXTERNAL_REFERENCE_FOUND | Preprocessing notebook scan did not find explicit copied blog/Kaggle/GitHub code references. If a member used an unrecorded external source, add it before final submission. |
| Modeling reference code | PASS | 모델링 코드는 scikit-learn/LightGBM 공식 API 사용 중심이며, 인터넷 복사 코드에 의존한 부분은 확인되지 않는다. |

## Library and API References

- pandas documentation: https://pandas.pydata.org/docs/
- NumPy documentation: https://numpy.org/doc/
- scikit-learn documentation: https://scikit-learn.org/stable/
- `train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- `KFold`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
- `LinearRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- `Ridge`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- `RandomForestRegressor`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- `KMeans`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- `AgglomerativeClustering`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- `IsolationForest`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- `SimpleImputer`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- `PCA`: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- LightGBM documentation: https://lightgbm.readthedocs.io/
- joblib documentation: https://joblib.readthedocs.io/
- Matplotlib documentation: https://matplotlib.org/stable/
- seaborn documentation: https://seaborn.pydata.org/

## Citation Note for Report

보고서에는 "The GSM smartphone data source is Kaggle GSMArena Mobile Phone Devices by Mohit Sainani, originally extracted from GSMArena. All external library APIs were used according to the official documentation listed above."라고 명시한다.
"""
    (DOC_DIR / "09_SOURCE_CITATION_CHECKLIST.md").write_text(citation_checklist, encoding="utf-8")

    presentation_outline = """# 10. Final Presentation Outline

과제 설명 PDF의 PPT presentation 조건을 채우기 위한 발표 구성이다. 실제 PPT 제출본은 `presentation/DS_Team7_GSM_Final_Presentation.pptx`에 생성되어 있고, 동일 내용의 백업 초안은 `presentation/DS_Team7_GSM_Final_Presentation_Draft.pptx`에 남겨 두었다.

## Slide 1. Project Title

- Prediction of Profit Cost for Company & Device Recommendation for Customer
- Team 7 members
- 기업용 적정가 예측 + 소비자용 예산 기반 추천 시스템

## Slide 2. Motivation and Objective

- 스마트폰 하드웨어 스펙과 브랜드 인지도가 가격에 미치는 영향 분석
- 기업 관점: 객관적 가격 가이드
- 소비자 관점: 예산 내 최적 가성비 모델 추천

## Slide 3. Dataset

- Raw GSM data: 10,679 rows x 86 columns
- Numerical data: price, RAM, storage, camera MP, battery, display size, launch year
- Categorical data: OEM/brand, OS family, display panel, battery type, launch status
- Dirty data: missing values, mixed unit strings, text-based hardware specs, price strings

## Slide 4. Preprocessing and Feature Engineering

- Missing value handling
- Price parsing to EUR
- Hardware text parsing
- Brand grouping
- OneHotEncoder and StandardScaler
- Output files in `content/`

## Slide 5. Regression Price Prediction

- Models: Linear Regression, Ridge, Random Forest, LightGBM
- Best model: LightGBM with log price target
- Holdout MAE: 51.73 EUR
- 5-fold CV MAE: 44.95 EUR
- Explain why MAE/MASE are useful for price guide

## Slide 6. Price Drivers and Brand Premium

- Top hardware features from feature importance
- Brand one-hot features and adjusted brand premium
- Business interpretation: same-spec phones can differ by brand signal

## Slide 7. Clustering Segmentation

- KMeans k=3
- Entry, Mid-range, Flagship
- Segment counts and average price/spec summary

## Slide 8. Value Outlier Detection

- PPR = spec score / price
- Segment Z-score
- IsolationForest
- Expected market price underpricing
- Final value outliers: 170 models

## Slide 9. Budget Recommendation

- User enters budget
- Filter price <= budget
- Rank by PPR, value score, outlier flag, spec score, lower price
- Constraint validation: all recommendations within budget and PPR descending

## Slide 10. Two-Way Solution Demo

- `two_way_solution.py --mode demo`
- Business price guide examples
- User recommendation examples

## Slide 11. Evaluation and Limitations

- Regression metrics, CV, cluster quality, recommendation validation
- Limitations: EUR prices, older/newer phones mixed, parsed hardware specs, regional/channel prices not reflected

## Slide 12. Teamwork, Learning, and References

- Contribution percentage table
- Each member learned writeup summary
- Dataset source and official library documentation citations
"""
    (DOC_DIR / "10_FINAL_PRESENTATION_OUTLINE.md").write_text(presentation_outline, encoding="utf-8")

    final_audit = f"""# 최종 모델링 감사 리포트

## 점검 기준
- `MODELING_PROMPT.md`
- `TermProject_specification_c1_0507.pdf`
- `Term_Project_Proposal.pdf` / `DS_Proposal_한글화.pdf`
- `content/` 정리 CSV 5개와 `inspection_data/` 자료

## 결과
- 최종 회귀 모델: {best['model']} ({best['target_mode']})
- Holdout MAE: {best['MAE']:.2f} EUR
- Holdout R2: {best['R2']:.4f}
- 세그먼트 수: {segmented['segment'].nunique()}
- 가성비 이상치 수: {int(value_df['is_value_outlier'].sum())}

## 평가 리스크
모델링 파트 기준 필수 조건은 충족했다. 다만 팀 최종 제출물에는 팀원별 기여도와 학습 회고가 별도로 필요하다.
"""
    (MODELING_DIR / "FINAL_MODELING_AUDIT.md").write_text(final_audit, encoding="utf-8")


def main() -> None:
    """Run every modeling task end to end."""
    ensure_dirs()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    data_source = find_data_source()
    print(f"[INFO] using data source: {data_source}")

    price_model_df = read_modeling_csv(data_source, PRICE_MODEL_CSV)
    price_raw_df = read_modeling_csv(data_source, PRICE_RAW_CSV)
    reco_model_df = read_modeling_csv(data_source, RECO_MODEL_CSV)
    reco_raw_df = read_modeling_csv(data_source, RECO_RAW_CSV)

    data_summary = pd.DataFrame(
        [
            {
                "file": PRICE_MODEL_CSV,
                "rows": len(price_model_df),
                "cols": price_model_df.shape[1],
                "null_total": int(price_model_df.isna().sum().sum()),
                "object_cols": len(price_model_df.select_dtypes(include=["object", "string"]).columns),
            },
            {
                "file": PRICE_RAW_CSV,
                "rows": len(price_raw_df),
                "cols": price_raw_df.shape[1],
                "null_total": int(price_raw_df.isna().sum().sum()),
                "object_cols": len(price_raw_df.select_dtypes(include=["object", "string"]).columns),
            },
            {
                "file": RECO_MODEL_CSV,
                "rows": len(reco_model_df),
                "cols": reco_model_df.shape[1],
                "null_total": int(reco_model_df.isna().sum().sum()),
                "object_cols": len(reco_model_df.select_dtypes(include=["object", "string"]).columns),
            },
            {
                "file": RECO_RAW_CSV,
                "rows": len(reco_raw_df),
                "cols": reco_raw_df.shape[1],
                "null_total": int(reco_raw_df.isna().sum().sum()),
                "object_cols": len(reco_raw_df.select_dtypes(include=["object", "string"]).columns),
            },
        ]
    )
    data_summary.to_csv(OUTPUT_DIR / "input_data_summary.csv", index=False)

    leakage_cols = [
        "price_tier",
        "value_score",
        "spec_score_0_100",
        "price_per_ram_gb",
        "price_per_storage_gb",
        "battery_per_eur",
        "oem",
        "model",
    ]
    leakage_report = pd.DataFrame(
        {
            "column": leakage_cols,
            "present_in_price_model": [col in price_model_df.columns for col in leakage_cols],
        }
    )
    leakage_report.to_csv(OUTPUT_DIR / "regression_dropped_columns.csv", index=False)

    regression = train_and_evaluate_regression(price_model_df)
    plot_regression_metrics(regression.cv_metrics)
    importance_df, _ = feature_importance_analysis(price_model_df, regression.feature_names)
    adjusted_brand = brand_premium_analysis(price_model_df, price_raw_df)
    create_model_reference(price_model_df, price_raw_df)
    segmented = cluster_segments(reco_raw_df)
    value_df, recommendations = value_outliers_and_recommendations(segmented)
    save_price_driver_summary(importance_df)

    # two_way_solution.py is created before this script is run; demo outputs are
    # generated after model artifacts exist.
    make_business_demo()
    from two_way_solution import TwoWaySmartphoneSolution

    solution = TwoWaySmartphoneSolution(root_dir=ROOT_DIR)
    demo_recs = []
    demo_scenarios = [
        {"scenario": "student_balanced", "budget_eur": 200, "segment": None},
        {"scenario": "worker_midrange", "budget_eur": 400, "segment": "Mid-range"},
        {"scenario": "premium_flagship", "budget_eur": 800, "segment": "Flagship"},
        {"scenario": "entry_second_phone", "budget_eur": 150, "segment": "Entry"},
    ]
    for sc in demo_scenarios:
        rec = solution.recommend_for_user(sc["budget_eur"], top_n=8, segment=sc["segment"])
        rec.insert(0, "scenario", sc["scenario"])
        rec.insert(1, "budget_eur", sc["budget_eur"])
        demo_recs.append(rec)
    pd.concat(demo_recs, ignore_index=True).to_csv(
        OUTPUT_DIR / "two_way_user_recommendations.csv", index=False
    )

    write_docs(data_source, regression, importance_df, adjusted_brand, segmented, value_df)

    print("[DONE] GSM modeling pipeline complete")
    print(regression.metrics.head().to_string(index=False))
    print(pd.read_csv(OUTPUT_DIR / "cluster_segment_summary.csv").to_string(index=False))
    print(pd.read_csv(OUTPUT_DIR / "value_outlier_validation_summary.csv").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run full modeling pipeline")
    args = parser.parse_args()
    main()
