# Module: config
# Purpose: Paths, split defaults, and leakage-column guards for the Regression ensemble workspace.

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT_DIR = Path(__file__).resolve().parents[1]
REGRESSION_DIR = ROOT_DIR / "Regression"
OUTPUT_DIR = REGRESSION_DIR / "outputs"
MODEL_DIR = REGRESSION_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

CONTENT_DIR = ROOT_DIR / "content"
PRICE_MODEL_CSV = "gsm_processed_all(price_prediction).csv"
TARGET_COL = "target_price_eur"

# Columns that must never enter price regression features.
LEAKAGE_COLS = {
    "price_tier",
    "value_score",
    "spec_score_0_100",
    "price_per_ram_gb",
    "price_per_storage_gb",
    "battery_per_eur",
    "oem",
    "model",
}
