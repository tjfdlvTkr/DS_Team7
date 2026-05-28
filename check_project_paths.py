"""Check that the DS_Team7 project paths needed by Colab/modeling exist."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_PATHS = [
    "Preprocessing/GSM_전처리+피처엔지니어링.ipynb",
    "content/gsm.csv",
    "content/gsm_processed_all(price_prediction).csv",
    "content/gsm_processed(price_prediction).csv",
    "content/gsm_processed_all(recommendation).csv",
    "content/gsm_processed(recommendation).csv",
    "inspection_data/SUMMARY.md",
    "modeling/GSM_modeling_colab.ipynb",
    "modeling/GSM__모델링.ipynb",
    "modeling/run_modeling.py",
    "modeling/two_way_solution.py",
]


def main() -> None:
    print(f"PROJECT_ROOT={ROOT}")
    missing: list[str] = []
    for rel in REQUIRED_PATHS:
        path = ROOT / rel
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:7} {rel}")
        if not path.exists():
            missing.append(rel)
    if missing:
        raise SystemExit(
            "Missing required project paths. Use the GitHub root layout with "
            "DS_Team7/content and DS_Team7/modeling at the same level."
        )
    print("PATH_CHECK_PASS")


if __name__ == "__main__":
    main()
