# =============================================================================
# phase3_data_preparation.py — Data Preparation & Cleaning (CRISP-DM Phase 3)
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# Steps performed:
#   1. Remove leakage and metadata columns
#   2. Encode categorical features with LabelEncoder
#   3. Standardise numerical features for Logistic Regression (StandardScaler)
#   4. Encode binary target variable (Yes=1, No=0)
#   5. Stratified train/test split (80/20)
#   6. Save prepared splits to outputs/ for reproducibility
#
# Returns: X_train, X_test, y_train, y_test, feature_names, scaler
# =============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from config import (DATA_PATH, OUTPUT_DIR, NUMERICAL_FEATURES,
                    CATEGORICAL_FEATURES, LEAKAGE_COLS, TARGET_COL,
                    POSITIVE_CLASS, TEST_SIZE, RANDOM_STATE, MODEL_DIR)


# ── Step 1: Load ──────────────────────────────────────────────────────────────
def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Raw dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ── Step 2: Drop Leakage and Metadata Columns ─────────────────────────────────
def drop_leakage(df):
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"\n[Step 2] Dropped {len(cols_to_drop)} leakage/metadata columns:")
    for c in cols_to_drop:
        print(f"         - {c}")
    print(f"         Remaining columns: {df.shape[1]}")
    return df


# ── Step 3: Handle Missing Values ─────────────────────────────────────────────
def handle_missing(df):
    before = df.isnull().sum().sum()
    # Numerical: fill with column median
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [Missing] {col}: filled {df[col].isnull().sum()} NaN with median={median_val:.2f}")

    # Categorical: fill with mode
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [Missing] {col}: filled NaN with mode='{mode_val}'")

    after = df.isnull().sum().sum()
    print(f"\n[Step 3] Missing values: {before} → {after}")
    return df


# ── Step 4: Encode Categorical Features ──────────────────────────────────────
def encode_categoricals(df):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        new_col = col + "_enc"
        df[new_col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        unique_vals = list(le.classes_)
        print(f"  [Encode] {col} → {new_col}  ({len(unique_vals)} classes: {unique_vals[:5]}{'...' if len(unique_vals)>5 else ''})")
    print(f"\n[Step 4] Encoded {len(cat_cols)} categorical features")

    # Save encoders for deployment
    enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"         Label encoders saved → {enc_path}")
    return df, encoders


# ── Step 5: Build Final Feature Matrix ────────────────────────────────────────
def build_feature_matrix(df):
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    enc_cols = [c + "_enc" for c in CATEGORICAL_FEATURES if c + "_enc" in df.columns]
    feature_names = num_cols + enc_cols

    X = df[feature_names].copy()
    y = (df[TARGET_COL] == POSITIVE_CLASS).astype(int)

    print(f"\n[Step 5] Feature matrix shape: {X.shape}")
    print(f"         Numerical features  : {len(num_cols)}")
    print(f"         Encoded categoricals: {len(enc_cols)}")
    print(f"         Total features      : {len(feature_names)}")
    print(f"         Target distribution : {y.value_counts().to_dict()}")
    print(f"         Positive class (flood=Yes): {y.mean()*100:.1f}%")
    return X, y, feature_names


# ── Step 6: Train/Test Split ──────────────────────────────────────────────────
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y           # preserves class ratio in both splits
    )
    print(f"\n[Step 6] Stratified train/test split (80/20):")
    print(f"         Train set : {X_train.shape[0]:,} records  "
          f"(flood={y_train.sum():,} = {y_train.mean()*100:.1f}%)")
    print(f"         Test set  : {X_test.shape[0]:,} records  "
          f"(flood={y_test.sum():,} = {y_test.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ── Step 7: Standardise for Logistic Regression ───────────────────────────────
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    # Save scaler for deployment
    sc_path = os.path.join(MODEL_DIR, "standard_scaler.pkl")
    with open(sc_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n[Step 7] StandardScaler fitted and saved → {sc_path}")
    return X_train_sc, X_test_sc, scaler


# ── Step 8: Save Prepared Data ────────────────────────────────────────────────
def save_splits(X_train, X_test, y_train, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.assign(target=y_train.values).to_csv(
        os.path.join(OUTPUT_DIR, "train_set.csv"), index=False)
    X_test.assign(target=y_test.values).to_csv(
        os.path.join(OUTPUT_DIR, "test_set.csv"), index=False)
    print(f"\n[Step 8] Prepared datasets saved to: {OUTPUT_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────
def prepare_data(save=True):
    print("\n" + "=" * 60)
    print("PHASE 3 — DATA PREPARATION & CLEANING")
    print("=" * 60)

    df = load_raw()
    df = drop_leakage(df)
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)
    X, y, feature_names = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    if save:
        save_splits(X_train, X_test, y_train, y_test)

    print("\n[Phase 3 complete] Data preparation done.")
    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc,
            feature_names, scaler, encoders)


if __name__ == "__main__":
    prepare_data()
