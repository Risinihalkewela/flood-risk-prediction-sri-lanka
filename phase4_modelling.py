# =============================================================================
# phase4_modelling.py — Model Training (CRISP-DM Phase 4)
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# Trains four classifiers with 5-fold stratified cross-validation:
#   1. Logistic Regression  (baseline, linear)
#   2. Decision Tree        (interpretable, single tree)
#   3. Random Forest        (ensemble, bagging)
#   4. Gradient Boosting    (ensemble, boosting — best performer)
#
# Outputs:
#   - Trained model objects saved as .pkl files
#   - Cross-validation scores printed and saved
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

from config import (MODEL_DIR, CV_FOLDS, RANDOM_STATE,
                    LR_PARAMS, DT_PARAMS, RF_PARAMS, GB_PARAMS)


# ── Define Models ─────────────────────────────────────────────────────────────
def get_models():
    """
    Returns an ordered dict of model_name → (model_instance, needs_scaling)
    needs_scaling=True means we pass the StandardScaler-transformed X to this model
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(**LR_PARAMS),
            True   # requires StandardScaler
        ),
        "Decision Tree": (
            DecisionTreeClassifier(**DT_PARAMS),
            False
        ),
        "Random Forest": (
            RandomForestClassifier(**RF_PARAMS),
            False
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(**GB_PARAMS),
            False
        ),
    }
    return models


# ── Cross-Validation ──────────────────────────────────────────────────────────
def cross_validate_models(models, X_train, X_train_sc, y_train):
    """
    Run 5-fold stratified CV on each model.
    Reports AUC-ROC, Accuracy, F1 (macro) for each fold.
    """
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION ({CV_FOLDS}-fold Stratified)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, (model, scaled) in models.items():
        X = X_train_sc if scaled else X_train
        scores = cross_validate(
            model, X, y_train, cv=cv,
            scoring=["roc_auc", "accuracy", "f1"],
            return_train_score=False,
            n_jobs=-1
        )
        cv_results[name] = {
            "auc_mean":  scores["test_roc_auc"].mean(),
            "auc_std":   scores["test_roc_auc"].std(),
            "acc_mean":  scores["test_accuracy"].mean(),
            "f1_mean":   scores["test_f1"].mean(),
        }
        print(f"\n  {name}")
        print(f"    ROC-AUC : {scores['test_roc_auc'].mean():.4f}  ± {scores['test_roc_auc'].std():.4f}")
        print(f"    Accuracy: {scores['test_accuracy'].mean():.4f}  ± {scores['test_accuracy'].std():.4f}")
        print(f"    F1 Score: {scores['test_f1'].mean():.4f}  ± {scores['test_f1'].std():.4f}")
        print(f"    Per-fold AUC: {[round(s,4) for s in scores['test_roc_auc']]}")

    return cv_results


# ── Train Final Models ────────────────────────────────────────────────────────
def train_models(models, X_train, X_train_sc, y_train):
    """
    Fit each model on the full training set and save to disk.
    Returns dict of name → fitted model.
    """
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODELS ON FULL TRAINING SET")
    print("=" * 60)

    trained = {}
    for name, (model, scaled) in models.items():
        X = X_train_sc if scaled else X_train
        print(f"\n  Fitting: {name} ...")
        model.fit(X, y_train)
        trained[name] = (model, scaled)

        # Save model
        safe_name = name.lower().replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"    Saved → {model_path}")

        # Decision Tree — print tree depth info
        if hasattr(model, "get_depth"):
            print(f"    Tree depth  : {model.get_depth()}")
            print(f"    Leaf nodes  : {model.get_n_leaves()}")

        # Random Forest — print OOB hint
        if hasattr(model, "n_estimators"):
            print(f"    n_estimators: {model.n_estimators}")

    print("\n[Phase 4 complete] All models trained and saved.")
    return trained


# ── Learning Curve (optional diagnostic) ─────────────────────────────────────
def print_model_complexity(trained, X_train, X_train_sc, y_train, X_test, X_test_sc, y_test):
    """
    Compare train vs test accuracy to diagnose overfitting/underfitting.
    """
    from sklearn.metrics import accuracy_score
    print("\n" + "=" * 60)
    print("TRAIN vs TEST ACCURACY (Bias-Variance Check)")
    print("=" * 60)
    print(f"  {'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for name, (model, scaled) in trained.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test
        train_acc = accuracy_score(y_train, model.predict(Xtr))
        test_acc  = accuracy_score(y_test,  model.predict(Xte))
        gap = train_acc - test_acc
        flag = " ← OVERFIT" if gap > 0.05 else ""
        print(f"  {name:<25} {train_acc:>10.4f} {test_acc:>10.4f} {gap:>10.4f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_modelling(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    print("\n" + "=" * 60)
    print("PHASE 4 — MODELLING")
    print("=" * 60)

    models = get_models()
    cv_results = cross_validate_models(models, X_train, X_train_sc, y_train)
    trained    = train_models(models, X_train, X_train_sc, y_train)
    print_model_complexity(trained, X_train, X_train_sc, y_train,
                           X_test, X_test_sc, y_test)
    return trained, cv_results


if __name__ == "__main__":
    # Run standalone — imports Phase 3
    from phase3_data_preparation import prepare_data
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     feature_names, scaler, encoders) = prepare_data(save=False)

    run_modelling(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)
