"""
Project-wide configuration — edit this file to control which dataset is used
across all notebooks.

MODE options:
  "FULL"   — full combined dataset (~1.5M rows, 60-90 min runtime)
  "TEST"   — stratified subsample (~44k rows, 2–5 min runtime)
  "CUSTOM" — use CUSTOM_PATH below (any CSV with the same column schema)
"""

MODE = "TEST"   # <-- change this: "FULL" | "TEST" | "CUSTOM"

CUSTOM_PATH = ""   # only used when MODE = "CUSTOM"

# ── Resolved paths (do not edit below) ───────────────────────────────────────
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

_PATHS = {
    "FULL":   os.path.join(_ROOT, "data", "Dataset", "Combined Data", "Combined_dataset.csv"),
    "TEST":   os.path.join(_ROOT, "test_data", "Combined_dataset_test.csv"),
}

_IRELAND_PATHS = {
    "FULL":   os.path.join(_ROOT, "data", "Dataset", "Country-Wise Data", "Ireland_dataset.csv"),
    "TEST":   os.path.join(_ROOT, "test_data", "Ireland_dataset_test.csv"),
}

if MODE == "CUSTOM":
    if not CUSTOM_PATH:
        raise ValueError("config.py: MODE='CUSTOM' but CUSTOM_PATH is empty.")
    DATA_PATH    = CUSTOM_PATH
    IRELAND_PATH = CUSTOM_PATH
elif MODE in _PATHS:
    DATA_PATH    = _PATHS[MODE]
    IRELAND_PATH = _IRELAND_PATHS[MODE]
else:
    raise ValueError(f"config.py: unknown MODE '{MODE}'. Use 'FULL', 'TEST', or 'CUSTOM'.")

# ── Shared experiment settings ────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(_ROOT, "output")
FEATURES = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
]
TARGET      = "CCME_Values"
RANDOM_SEED = 42
TEST_SIZE   = 0.1
VAL_SIZE    = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)