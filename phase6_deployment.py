# =============================================================================
# phase6_deployment.py — Deployment & Inference (CRISP-DM Phase 6)
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# Provides a deployment-ready inference pipeline:
#   - load_best_model()  — load saved Gradient Boosting model from disk
#   - predict_single()   — predict flood risk for one location
#   - predict_batch()    — predict for a CSV of new locations
#   - generate_risk_map() — visualise flood probability across all records
#   - risk_report()       — per-district summary report
#
# Usage:
#   python phase6_deployment.py --input new_locations.csv --output predictions.csv
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from config import (MODEL_DIR, OUTPUT_DIR, CHART_DIR, DATA_PATH,
                    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
                    TARGET_COL, POSITIVE_CLASS,
                    DEPLOYED_THRESHOLD, DEFAULT_THRESHOLD,
                    BLUE, RED, GREEN, CHART_DPI)


# ── Load Saved Model & Preprocessors ─────────────────────────────────────────
def load_pipeline():
    """Load the best model (Gradient Boosting) and preprocessing objects."""
    model_path = os.path.join(MODEL_DIR, "gradient_boosting.pkl")
    enc_path   = os.path.join(MODEL_DIR, "label_encoders.pkl")
    sc_path    = os.path.join(MODEL_DIR, "standard_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run phase4_modelling.py first to train and save the model."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[Load] Gradient Boosting model loaded from: {model_path}")

    encoders = {}
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)
        print(f"[Load] Label encoders loaded from: {enc_path}")

    scaler = None
    if os.path.exists(sc_path):
        with open(sc_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"[Load] StandardScaler loaded from: {sc_path}")

    return model, encoders, scaler


# ── Preprocess New Data ────────────────────────────────────────────────────────
def preprocess_input(df_raw, encoders):
    """
    Apply the same preprocessing pipeline used during training.
    Handles unseen categorical values gracefully.
    """
    df = df_raw.copy()

    # Encode categoricals using saved encoders
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            # Fill with a default / unknown value
            df[col] = "Unknown"
        enc_col = col + "_enc"
        le = encoders.get(col)
        if le is not None:
            # Handle unseen labels
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[enc_col] = le.transform(df[col].astype(str))
        else:
            # Fallback: integer encoding on the fly
            df[enc_col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Build feature matrix
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    enc_cols = [c + "_enc" for c in CATEGORICAL_FEATURES if c + "_enc" in df.columns]
    feature_names = num_cols + enc_cols

    # Fill any missing numerical values with 0
    X = df[feature_names].fillna(0)
    return X, feature_names


# ── Single-Record Prediction ──────────────────────────────────────────────────
def predict_single(record: dict, threshold=DEPLOYED_THRESHOLD):
    """
    Predict flood risk for a single location.

    Args:
        record: dict with feature values, e.g.:
            {
                "elevation_m": 45,
                "distance_to_river_m": 300,
                "rainfall_7d_mm": 180,
                "monthly_rainfall_mm": 350,
                "drainage_index": 0.25,
                "ndvi": 0.3,
                "ndwi": 0.15,
                "historical_flood_count": 2,
                "infrastructure_score": 30,
                "nearest_hospital_km": 5,
                "nearest_evac_km": 3,
                "population_density_per_km2": 800,
                "built_up_percent": 20,
                "district": "Kegalle",
                "landcover": "Agriculture",
                "soil_type": "Clay",
                "water_supply": "Municipal",
                "electricity": "Grid",
                "road_quality": "Poor (unpaved)",
                "urban_rural": "Rural"
            }
        threshold: decision threshold (default 0.35 for early warning)

    Returns:
        dict with prediction and probability
    """
    model, encoders, _ = load_pipeline()
    df_input = pd.DataFrame([record])
    X, _ = preprocess_input(df_input, encoders)

    prob      = model.predict_proba(X)[0, 1]
    predicted = int(prob >= threshold)
    label     = "FLOOD EXPECTED" if predicted == 1 else "No Flood"
    risk_cat  = (
        "HIGH"   if prob >= 0.65 else
        "MEDIUM" if prob >= 0.35 else
        "LOW"
    )

    result = {
        "flood_probability":  round(float(prob), 4),
        "prediction":         label,
        "risk_category":      risk_cat,
        "threshold_used":     threshold,
    }

    print(f"\n{'='*45}")
    print(f"  FLOOD RISK PREDICTION")
    print(f"{'='*45}")
    print(f"  Flood Probability : {prob:.1%}")
    print(f"  Prediction        : {label}")
    print(f"  Risk Category     : {risk_cat}")
    print(f"  Threshold Used    : {threshold}")
    print(f"{'='*45}")
    return result


# ── Batch Prediction ──────────────────────────────────────────────────────────
def predict_batch(input_path, output_path=None, threshold=DEPLOYED_THRESHOLD):
    """
    Run flood risk predictions on a CSV file of new locations.
    Saves results CSV with flood_probability and prediction columns appended.
    """
    model, encoders, _ = load_pipeline()
    df = pd.read_csv(input_path)
    print(f"\n[Batch] Loaded {len(df):,} records from: {input_path}")

    X, feature_names = preprocess_input(df, encoders)
    proba   = model.predict_proba(X)[:, 1]
    predict = (proba >= threshold).astype(int)

    df["flood_probability"] = proba.round(4)
    df["flood_prediction"]  = predict
    df["flood_label"]       = df["flood_prediction"].map({1: "FLOOD", 0: "No Flood"})
    df["risk_category"]     = pd.cut(
        df["flood_probability"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "batch_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"[Batch] Predictions saved → {output_path}")

    # Summary
    flood_count = predict.sum()
    print(f"\n[Batch Summary]")
    print(f"  Total records    : {len(df):,}")
    print(f"  Flood predicted  : {flood_count:,}  ({flood_count/len(df)*100:.1f}%)")
    print(f"  No flood         : {len(df)-flood_count:,}  ({(len(df)-flood_count)/len(df)*100:.1f}%)")
    return df


# ── Risk Map Visualisation ────────────────────────────────────────────────────
def generate_risk_map(input_path=DATA_PATH, threshold=DEPLOYED_THRESHOLD):
    """
    Generate a scatter-plot flood risk map using lat/lon.
    Colour intensity = predicted flood probability.
    """
    if not os.path.exists(input_path):
        print(f"Data not found: {input_path}")
        return

    model, encoders, _ = load_pipeline()
    df = pd.read_csv(input_path)

    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("[Risk Map] latitude/longitude columns not found — skipping map.")
        return

    X, _ = preprocess_input(df, encoders)
    proba = model.predict_proba(X)[:, 1]
    df["flood_probability"] = proba

    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(
        df["longitude"], df["latitude"],
        c=df["flood_probability"],
        cmap="RdYlGn_r",   # green = low risk, red = high risk
        s=5, alpha=0.6, linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Predicted Flood Probability", fontsize=11)

    ax.set_title("Sri Lanka — Predicted Flood Risk Map", fontsize=15, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude",  fontsize=11)

    # Overlay high-risk points
    high_risk = df[df["flood_probability"] >= threshold]
    ax.scatter(high_risk["longitude"], high_risk["latitude"],
               c="red", s=15, alpha=0.8, label=f"High Risk (≥{threshold})",
               linewidths=0.3, edgecolors="black")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(CHART_DIR, "15_flood_risk_map.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"[Risk Map] Saved → {out}")


# ── District Risk Report ──────────────────────────────────────────────────────
def risk_report(input_path=DATA_PATH, threshold=DEPLOYED_THRESHOLD):
    """
    Generate a per-district flood risk summary report.
    """
    if not os.path.exists(input_path):
        print(f"Data not found: {input_path}")
        return

    model, encoders, _ = load_pipeline()
    df = pd.read_csv(input_path)
    X, _ = preprocess_input(df, encoders)
    proba = model.predict_proba(X)[:, 1]
    df["flood_probability"] = proba
    df["flood_predicted"]   = (proba >= threshold).astype(int)

    if "district" not in df.columns:
        print("[Report] district column not found.")
        return

    report = df.groupby("district").agg(
        total_locations = ("flood_probability", "count"),
        avg_flood_prob  = ("flood_probability", "mean"),
        high_risk_count = ("flood_predicted",   "sum"),
    ).reset_index()
    report["high_risk_pct"] = (report["high_risk_count"] / report["total_locations"] * 100).round(1)
    report = report.sort_values("avg_flood_prob", ascending=False)

    print("\n" + "=" * 65)
    print("PER-DISTRICT FLOOD RISK REPORT")
    print("=" * 65)
    print(report.to_string(index=False))

    report_path = os.path.join(OUTPUT_DIR, "district_risk_report.csv")
    report.to_csv(report_path, index=False)
    print(f"\n[Report] Saved → {report_path}")

    # Chart
    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(report["district"], report["avg_flood_prob"] * 100,
                   color=[RED if p >= threshold * 100 else BLUE
                          for p in report["avg_flood_prob"] * 100],
                   edgecolor="white")
    ax.axvline(threshold * 100, color="black", linestyle="--", lw=2,
               label=f"Alert Threshold ({threshold*100:.0f}%)")
    ax.set_xlabel("Average Predicted Flood Probability (%)", fontsize=12)
    ax.set_title("Per-District Average Flood Probability", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "16_district_risk_report.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"[Report] District chart saved → {out}")
    return report


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood Risk Prediction — Deployment Script")
    parser.add_argument("--mode",   choices=["single", "batch", "map", "report"], default="report")
    parser.add_argument("--input",  default=DATA_PATH, help="Path to input CSV")
    parser.add_argument("--output", default=None,       help="Path to output predictions CSV")
    parser.add_argument("--threshold", type=float, default=DEPLOYED_THRESHOLD)
    args = parser.parse_args()

    if args.mode == "single":
        # Demo single prediction
        sample = {
            "elevation_m": 30, "distance_to_river_m": 250,
            "rainfall_7d_mm": 210, "monthly_rainfall_mm": 400,
            "drainage_index": 0.2, "ndvi": 0.2, "ndwi": 0.35,
            "historical_flood_count": 3, "infrastructure_score": 25,
            "nearest_hospital_km": 8, "nearest_evac_km": 4,
            "population_density_per_km2": 1200, "built_up_percent": 30,
            "district": "Kegalle", "landcover": "Agriculture",
            "soil_type": "Clay", "water_supply": "Municipal",
            "electricity": "Grid", "road_quality": "Poor (unpaved)",
            "urban_rural": "Rural"
        }
        predict_single(sample, threshold=args.threshold)

    elif args.mode == "batch":
        predict_batch(args.input, args.output, args.threshold)

    elif args.mode == "map":
        generate_risk_map(args.input, args.threshold)

    elif args.mode == "report":
        risk_report(args.input, args.threshold)
        generate_risk_map(args.input, args.threshold)
