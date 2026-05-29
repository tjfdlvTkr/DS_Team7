"""Base regressors for ensemble (GPU-aware when available)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from gpu import DevicePlan

RANDOM_STATE = 42


def make_ridge() -> Ridge:
    return Ridge(alpha=1.0, random_state=RANDOM_STATE)


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_hist_gradient_boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=48,
        random_state=RANDOM_STATE,
    )


def make_lightgbm(plan: DevicePlan) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is required. pip install lightgbm") from exc

    params: dict[str, Any] = dict(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    if plan.use_gpu:
        params["device"] = "gpu"
    else:
        params["device"] = "cpu"
    return LGBMRegressor(**params)


def make_xgboost(plan: DevicePlan) -> Any | None:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None

    params: dict[str, Any] = dict(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    if plan.use_gpu:
        params["device"] = "cuda"
    return XGBRegressor(**params)


def make_catboost(plan: DevicePlan) -> Any | None:
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return None

    params: dict[str, Any] = dict(
        iterations=600,
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    if plan.use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["task_type"] = "CPU"
    return CatBoostRegressor(**params)


def build_base_estimators(plan: DevicePlan) -> list[tuple[str, Any]]:
    """Return (name, estimator) pairs for stacking."""
    estimators: list[tuple[str, Any]] = [
        ("ridge", make_ridge()),
        ("random_forest", make_random_forest()),
        ("lightgbm", make_lightgbm(plan)),
    ]

    xgb = make_xgboost(plan)
    if xgb is not None:
        estimators.append(("xgboost", xgb))

    cat = make_catboost(plan)
    if cat is not None:
        estimators.append(("catboost", cat))

    return estimators
