# =============================================================================
# config.py — Global Configuration
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# This file centralises all constants, file paths, and model hyperparameters
# used throughout the project. Modify paths here to adapt to your environment.
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, r"C:\Users\HP\Documents\Semester 05\Data visualization and Story Telling\Assignment\Assignment 01\1\sri_lanka_flood_risk_dataset_25000.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
CHART_DIR   = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Target Variable ────────────────────────────────────────────────────────────
TARGET_COL  = "flood_occurrence_current_event"   # Binary: Yes / No
POSITIVE_CLASS = "Yes"                           # The minority / event class

# ── Features ──────────────────────────────────────────────────────────────────
# Numerical input features (no leakage)
NUMERICAL_FEATURES = [
    "elevation_m",
    "distance_to_river_m",
    "population_density_per_km2",
    "built_up_percent",
    "rainfall_7d_mm",
    "monthly_rainfall_mm",
    "drainage_index",
    "ndvi",
    "ndwi",
    "historical_flood_count",
    "infrastructure_score",
    "nearest_hospital_km",
    "nearest_evac_km",
]

# Categorical features to be label-encoded
CATEGORICAL_FEATURES = [
    "district",
    "landcover",
    "soil_type",
    "water_supply",
    "electricity",
    "road_quality",
    "urban_rural",
]

# Columns to drop — data leakage or metadata
LEAKAGE_COLS = [
    "inundation_area_sqm",   # Perfectly encodes flood occurrence
    "flood_risk_score",      # Outcome-derived continuous score
    "water_presence_flag",   # Derived from ndwi threshold
    "reason_not_good_to_live",
    "is_good_to_live",
    "record_id",
    "generation_date",
    "is_synthetic",
    "place_name",
    "latitude",
    "longitude",
]

# ── Model Hyperparameters ──────────────────────────────────────────────────────
LR_PARAMS = {
    "solver":       "lbfgs",
    "max_iter":     1000,
    "class_weight": "balanced",
    "C":            1.0,
    "random_state": 42,
}

DT_PARAMS = {
    "max_depth":        8,
    "criterion":        "gini",
    "class_weight":     "balanced",
    "min_samples_split": 2,
    "random_state":     42,
}

RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs":       -1,
}

GB_PARAMS = {
    "n_estimators":  200,
    "learning_rate": 0.1,
    "max_depth":     5,
    "subsample":     1.0,
    "random_state":  42,
}

# ── Train/Test Split ───────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Decision Threshold ─────────────────────────────────────────────────────────
# Default is 0.5; lower threshold improves flood recall at the cost of precision
DEFAULT_THRESHOLD  = 0.50
DEPLOYED_THRESHOLD = 0.35   # Recommended for early warning deployment

# ── Plotting Style ─────────────────────────────────────────────────────────────
CHART_DPI   = 150
BLUE        = "#1a6faf"
RED         = "#c0392b"
GREEN       = "#27ae60"
PURPLE      = "#8e44ad"
ORANGE      = "#e67e22"
