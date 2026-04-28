# =============================================================================
# phase2_eda.py — Exploratory Data Analysis (CRISP-DM Phase 2)
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# Performs thorough EDA including:
#   - Dataset overview and summary statistics
#   - Missing value analysis
#   - Target variable distribution
#   - Numerical feature distributions
#   - Pearson correlation heatmap
#   - Flood occurrence rate by district
#   - Box plots: key features vs flood occurrence
#   - Categorical feature value counts
#
# Outputs: charts saved to outputs/charts/
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from config import (DATA_PATH, CHART_DIR, NUMERICAL_FEATURES,
                    CATEGORICAL_FEATURES, TARGET_COL,
                    BLUE, RED, GREEN, ORANGE, CHART_DPI)

sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ── 2. Dataset Overview ───────────────────────────────────────────────────────
def dataset_overview(df):
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape            : {df.shape}")
    print(f"  Total cells      : {df.size:,}")
    print(f"  Duplicate rows   : {df.duplicated().sum()}")
    print(f"\nColumn dtypes:")
    print(df.dtypes.value_counts())
    print(f"\nFirst 3 rows preview:")
    print(df.head(3).to_string())


# ── 3. Summary Statistics ─────────────────────────────────────────────────────
def summary_statistics(df):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS — NUMERICAL FEATURES")
    print("=" * 60)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    desc = df[num_cols].describe().T
    desc["cv%"] = (desc["std"] / desc["mean"] * 100).round(1)
    print(desc[["count", "mean", "std", "min", "50%", "max", "cv%"]].to_string())


# ── 4. Missing Value Analysis ─────────────────────────────────────────────────
def missing_value_analysis(df):
    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    report = report[report["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    if report.empty:
        print("  No missing values found in any column.")
    else:
        print(report.to_string())

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    if not report.empty:
        report["Missing Count"].plot(kind="bar", ax=ax, color=RED, edgecolor="white")
        ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center",
                fontsize=16, color="green", transform=ax.transAxes)
        ax.set_title("Missing Values Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "01_missing_values.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")
    return report


# ── 5. Target Variable Distribution ──────────────────────────────────────────
def target_distribution(df):
    print("\n" + "=" * 60)
    print(f"TARGET VARIABLE: {TARGET_COL}")
    print("=" * 60)
    counts = df[TARGET_COL].value_counts()
    print(counts)
    print(f"  Class ratio (No:Yes) = {counts.get('No',0)/max(counts.get('Yes',1),1):.1f}:1")
    print(f"  Minority class %     = {counts.get('Yes',0)/len(df)*100:.1f}%")

    # Plot all 3 outcome variables together
    targets = [
        ("flood_occurrence_current_event", "Flood Occurrence"),
        ("is_good_to_live",                "Is Good to Live"),
        ("water_presence_flag",            "Water Presence Flag"),
    ]
    targets = [(c, t) for c, t in targets if c in df.columns]

    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, targets):
        vc = df[col].value_counts()
        bars = ax.bar(vc.index, vc.values, color=[BLUE, RED][: len(vc)],
                      edgecolor="white", width=0.5)
        for bar, cnt in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vc.values) * 0.02,
                    f"{cnt:,}\n({cnt/len(df)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(vc.values) * 1.25)
    plt.suptitle("Target Variable Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "02_target_distribution.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")


# ── 6. Numerical Feature Distributions ───────────────────────────────────────
def numerical_distributions(df):
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    n = len(num_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flat
    for ax, col in zip(axes, num_cols):
        data = df[col].dropna()
        ax.hist(data, bins=40, color=BLUE, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color=RED, linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color=GREEN, linestyle=":", linewidth=1.5,
                   label=f"Median: {data.median():.1f}")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    # Hide unused axes
    for ax in list(axes)[n:]:
        ax.set_visible(False)

    plt.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "03_numerical_distributions.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Numerical distributions chart saved → {out}")


# ── 7. Correlation Heatmap ────────────────────────────────────────────────────
def correlation_heatmap(df):
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    # Add flood_risk_score just for correlation analysis (will be dropped in modelling)
    extra = ["flood_risk_score", "historical_flood_count",
             "population_density_per_km2", "built_up_percent",
             "nearest_hospital_km", "nearest_evac_km", "inundation_area_sqm"]
    all_num = num_cols + [c for c in extra if c in df.columns and c not in num_cols]

    corr = df[all_num].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Pearson Correlation Matrix — Numerical Features",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "04_correlation_heatmap.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Correlation heatmap saved → {out}")

    # Print top correlations with target proxy (flood_risk_score)
    if "flood_risk_score" in corr.columns:
        top = corr["flood_risk_score"].drop("flood_risk_score").abs().sort_values(ascending=False)
        print("\n  Top correlations with flood_risk_score:")
        print(top.head(8).to_string())


# ── 8. Flood Occurrence by District ──────────────────────────────────────────
def flood_by_district(df):
    if "district" not in df.columns or TARGET_COL not in df.columns:
        return
    rate = df.groupby("district")[TARGET_COL].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100
    ).sort_values(ascending=False)

    print(f"\n  Flood occurrence rate by district (top 5):")
    print(rate.head(5).to_string())

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(rate.index, rate.values, color=BLUE, edgecolor="white")
    ax.set_title("Flood Occurrence Rate by District (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Flood Occurrence Rate (%)")
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "05_flood_by_district.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  District chart saved → {out}")


# ── 9. Box Plots: Features vs Flood Occurrence ───────────────────────────────
def boxplots_vs_target(df):
    key_features = [
        "rainfall_7d_mm", "monthly_rainfall_mm", "elevation_m",
        "distance_to_river_m", "drainage_index", "ndwi",
    ]
    key_features = [c for c in key_features if c in df.columns]

    n = len(key_features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flat

    for ax, feat in zip(axes, key_features):
        no_flood = df[df[TARGET_COL] == "No"][feat].dropna()
        flood    = df[df[TARGET_COL] == "Yes"][feat].dropna()
        bp = ax.boxplot([no_flood, flood], labels=["No Flood", "Flood"],
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5))
        bp["boxes"][0].set_facecolor(BLUE)
        bp["boxes"][1].set_facecolor(RED)
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")

    for ax in list(axes)[n:]:
        ax.set_visible(False)

    plt.suptitle("Key Features vs Flood Occurrence", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "06_boxplots_vs_flood.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Box plots chart saved → {out}")


# ── 10. Categorical Feature Value Counts ─────────────────────────────────────
def categorical_analysis(df):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    n = len(cat_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flat

    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts().head(10)
        vc.plot(kind="bar", ax=ax, color=BLUE, edgecolor="white")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    for ax in list(axes)[n:]:
        ax.set_visible(False)

    plt.suptitle("Categorical Feature Value Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "07_categorical_distributions.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Categorical chart saved → {out}")


# ── 11. Flood Risk Score Distribution by Flood Occurrence ─────────────────────
def risk_score_distribution(df):
    if "flood_risk_score" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("No", BLUE), ("Yes", RED)]:
        subset = df[df[TARGET_COL] == label]["flood_risk_score"].dropna()
        ax.hist(subset, bins=50, alpha=0.6, color=color, edgecolor="white",
                label=f"Flood = {label} (n={len(subset):,})")
    ax.set_title("Flood Risk Score Distribution by Flood Occurrence",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Flood Risk Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "08_risk_score_distribution.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Risk score distribution chart saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_eda():
    print("\n" + "=" * 60)
    print("PHASE 2 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_data()
    dataset_overview(df)
    summary_statistics(df)
    missing_value_analysis(df)
    target_distribution(df)
    numerical_distributions(df)
    correlation_heatmap(df)
    flood_by_district(df)
    boxplots_vs_target(df)
    categorical_analysis(df)
    risk_score_distribution(df)

    print("\n[Phase 2 complete] All EDA charts saved to:", CHART_DIR)
    return df


if __name__ == "__main__":
    run_eda()
