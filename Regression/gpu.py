"""GPU availability checks for tree boosting libraries."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DevicePlan:
    requested: str
    use_gpu: bool
    lightgbm_device: str
    xgboost_device: str
    catboost_task_type: str
    notes: list[str]


def _cuda_visible() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_device(requested: str = "auto") -> DevicePlan:
    """Resolve device plan: auto | gpu | cpu."""
    requested = requested.lower().strip()
    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("device must be one of: auto, gpu, cpu")

    notes: list[str] = []
    cuda_ok = _cuda_visible()
    if requested == "cpu":
        return DevicePlan("cpu", False, "cpu", "cpu", "CPU", ["forced CPU"])

    if requested == "gpu" and not cuda_ok:
        notes.append("GPU requested but CUDA not detected; falling back to CPU")
        return DevicePlan("gpu", False, "cpu", "cpu", "CPU", notes)

    if requested == "auto" and not cuda_ok:
        notes.append("CUDA not detected; using CPU")
        return DevicePlan("auto", False, "cpu", "cpu", "CPU", notes)

    notes.append("GPU mode enabled for LightGBM / XGBoost / CatBoost when installed")
    # Respect user GPU selection (e.g. RTX 5060 Ti); default device 0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return DevicePlan(requested if requested != "auto" else "gpu", True, "gpu", "cuda", "GPU", notes)
