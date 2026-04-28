# =============================================================================
# phase5_evaluation.py — Model Evaluation (CRISP-DM Phase 5)
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# Comprehensive evaluation of all trained models:
#   - Accuracy, ROC-AUC, Precision, Recall, F1 per class
#   - Confusion matrices (heatmaps)
#   - ROC curves (all models on one chart)
#   - Precision-Recall curves
#   - Feature importance (Random Forest & Gradient Boosting)
#   - Threshold analysis (effect of lowering decision threshold)
#   - Final model selection recommendation
#
# Outputs: charts saved to outputs/charts/, metrics to outputs/metrics.json
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, f1_score
)

from config import (CHART_DIR, OUTPUT_DIR, BLUE, RED, GREEN, PURPLE, ORANGE,
                    CHART_DPI, DEFAULT_THRESHOLD, DEPLOYED_THRESHOLD)

sns.set_theme(style="whitegrid", palette="muted")
COLORS = [BLUE, RED, GREEN, PURPLE]


# ── Predict with Threshold ────────────────────────────────────────────────────
def predict_with_threshold(model, X, threshold=DEFAULT_THRESHOLD):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba


# ── 1. Full Metrics Table ─────────────────────────────────────────────────────
def compute_all_metrics(trained, X_test, X_test_sc, y_test, threshold=DEFAULT_THRESHOLD):
    print("\n" + "=" * 60)
    print(f"EVALUATION METRICS (threshold = {threshold})")
    print("=" * 60)

    results = {}
    for name, (model, scaled) in trained.items():
        Xte = X_test_sc if scaled else X_test
        y_pred, y_prob = predict_with_threshold(model, Xte, threshold)

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        rep   = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        flood = rep.get("1", {})

        results[name] = {
            "accuracy":        round(acc,  4),
            "roc_auc":         round(auc,  4),
            "precision_flood": round(flood.get("precision", 0), 4),
            "recall_flood":    round(flood.get("recall",    0), 4),
            "f1_flood":        round(flood.get("f1-score",  0), 4),
            "f1_macro":        round(rep.get("macro avg", {}).get("f1-score", 0), 4),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model":  model,
            "scaled": scaled,
        }

        print(f"\n  {name}")
        print(f"    Accuracy        : {acc:.4f}")
        print(f"    ROC-AUC         : {auc:.4f}")
        print(f"    Flood Precision : {flood.get('precision',0):.4f}")
        print(f"    Flood Recall    : {flood.get('recall',0):.4f}")
        print(f"    Flood F1        : {flood.get('f1-score',0):.4f}")
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0,
                                    target_names=["No Flood", "Flood"]))

    return results


# ── 2. Model Comparison Bar Chart ─────────────────────────────────────────────
def plot_model_comparison(results, cv_results=None):
    model_names = list(results.keys())
    accs  = [results[m]["accuracy"]  for m in model_names]
    aucs  = [results[m]["roc_auc"]   for m in model_names]
    f1s   = [results[m]["f1_flood"]  for m in model_names]
    cvs   = [cv_results[m]["auc_mean"] for m in model_names] if cv_results else aucs

    x = np.arange(len(model_names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - 1.5*w, accs, w, label="Accuracy",    color="#2980b9", edgecolor="white")
    b2 = ax.bar(x - 0.5*w, aucs, w, label="ROC-AUC",     color="#27ae60", edgecolor="white")
    b3 = ax.bar(x + 0.5*w, f1s,  w, label="Flood F1",    color="#e67e22", edgecolor="white")
    b4 = ax.bar(x + 1.5*w, cvs,  w, label="CV AUC",      color="#8e44ad", edgecolor="white")

    for bar_group in [b1, b2, b3, b4]:
        for bar in bar_group:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Flood Occurrence Prediction",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "09_model_comparison.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved → {out}")


# ── 3. ROC Curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, res), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(
            [1]*len(res["y_prob"]),  # dummy — use y_test passed separately
            res["y_prob"]
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)

    # Redo with actual y_test
    ax.cla()
    for (name, res), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(res["_y_test"], res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "10_roc_curves.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves chart saved → {out}")


def plot_roc_curves_fixed(results, y_test):
    """Correct version — passes y_test explicitly."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, res), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "10_roc_curves.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves chart saved → {out}")


# ── 4. Confusion Matrices ─────────────────────────────────────────────────────
def plot_confusion_matrices(results, y_test):
    model_names = list(results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_test, results[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Flood", "Flood"],
                    yticklabels=["No Flood", "Flood"])
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Annotate TN/FP/FN/TP
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.12, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
                transform=ax.transAxes, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    out = os.path.join(CHART_DIR, "11_confusion_matrices.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrices chart saved → {out}")


# ── 5. Precision-Recall Curves ────────────────────────────────────────────────
def plot_precision_recall(results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", lw=1.2,
               label=f"No Skill Baseline ({baseline:.2f})")

    for (name, res), color in zip(results.items(), COLORS):
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        ap = average_precision_score(y_test, res["y_prob"])
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f"{name}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "12_precision_recall_curves.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Precision-Recall curves saved → {out}")


# ── 6. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(results, feature_names, top_n=15):
    imp_models = {
        name: res for name, res in results.items()
        if hasattr(res["model"], "feature_importances_")
    }
    if not imp_models:
        print("  No tree-based models found for feature importance.")
        return

    n = len(imp_models)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    if n == 1:
        axes = [axes]

    fig.suptitle("Feature Importance (Mean Decrease in Impurity)",
                 fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, imp_models.items()):
        fi = pd.Series(
            res["model"].feature_importances_, index=feature_names
        ).sort_values(ascending=True).tail(top_n)

        bars = ax.barh(fi.index, fi.values, color=BLUE, edgecolor="white")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature Importance", fontsize=11)
        for bar, val in zip(bars, fi.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(CHART_DIR, "13_feature_importance.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart saved → {out}")


# ── 7. Threshold Analysis ─────────────────────────────────────────────────────
def threshold_analysis(results, y_test, model_name="Gradient Boosting"):
    """
    Show how precision, recall, and F1 change as the decision threshold varies.
    """
    if model_name not in results:
        model_name = list(results.keys())[-1]

    y_prob = results[model_name]["y_prob"]
    thresholds = np.linspace(0.1, 0.9, 81)
    precisions, recalls, f1s, accs = [], [], [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0,
                     average="binary")  # F1
        r = accuracy_score(y_test, y_pred_t)
        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        rec  = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        f1s.append(p)
        accs.append(r)
        precisions.append(prec)
        recalls.append(rec)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, precisions, color=BLUE,   lw=2.5, label="Precision (Flood)")
    ax.plot(thresholds, recalls,    color=RED,    lw=2.5, label="Recall (Flood)")
    ax.plot(thresholds, f1s,        color=GREEN,  lw=2.5, label="F1 Score (Flood)")
    ax.plot(thresholds, accs,       color=ORANGE, lw=2,   linestyle="--", label="Overall Accuracy")

    ax.axvline(DEFAULT_THRESHOLD,  color="gray",  linestyle="--", lw=1.5, label=f"Default ({DEFAULT_THRESHOLD})")
    ax.axvline(DEPLOYED_THRESHOLD, color="black", linestyle=":",  lw=2,   label=f"Recommended ({DEPLOYED_THRESHOLD})")

    ax.set_xlabel("Decision Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Threshold Analysis — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out = os.path.join(CHART_DIR, "14_threshold_analysis.png")
    plt.savefig(out, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Threshold analysis chart saved → {out}")

    # Print key thresholds
    print(f"\n  Threshold effect on {model_name}:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for t in [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        idx = np.argmin(np.abs(thresholds - t))
        print(f"  {t:>10.2f} {precisions[idx]:>10.4f} {recalls[idx]:>10.4f} "
              f"{f1s[idx]:>10.4f} {accs[idx]:>10.4f}")


# ── 8. Save Metrics JSON ──────────────────────────────────────────────────────
def save_metrics(results, cv_results=None):
    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {
            "accuracy":        res["accuracy"],
            "roc_auc":         res["roc_auc"],
            "precision_flood": res["precision_flood"],
            "recall_flood":    res["recall_flood"],
            "f1_flood":        res["f1_flood"],
            "f1_macro":        res["f1_macro"],
        }
        if cv_results and name in cv_results:
            metrics_out[name]["cv_auc_mean"] = round(cv_results[name]["auc_mean"], 4)
            metrics_out[name]["cv_auc_std"]  = round(cv_results[name]["auc_std"],  4)

    path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics saved → {path}")
    return metrics_out


# ── 9. Final Recommendation ────────────────────────────────────────────────────
def print_recommendation(results, cv_results=None):
    print("\n" + "=" * 60)
    print("MODEL SELECTION RECOMMENDATION")
    print("=" * 60)

    # Score = 0.5 * AUC + 0.3 * Recall_flood + 0.2 * Accuracy
    scores = {}
    for name, res in results.items():
        score = 0.5 * res["roc_auc"] + 0.3 * res["recall_flood"] + 0.2 * res["accuracy"]
        scores[name] = score

    best = max(scores, key=scores.get)
    print(f"\n  Composite scoring (50% AUC + 30% Flood Recall + 20% Accuracy):")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {name:<25} {score:.4f}")
    print(f"\n  RECOMMENDED MODEL: {best}")
    print(f"    ROC-AUC         : {results[best]['roc_auc']:.4f}")
    print(f"    Accuracy        : {results[best]['accuracy']:.4f}")
    print(f"    Flood Recall    : {results[best]['recall_flood']:.4f}")
    print(f"    Flood F1        : {results[best]['f1_flood']:.4f}")
    print(f"\n  DEPLOYMENT NOTE: Lower decision threshold to {DEPLOYED_THRESHOLD}")
    print(f"  to improve Flood Recall for early warning applications.")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_evaluation(trained, X_test, X_test_sc, y_test, feature_names, cv_results=None):
    print("\n" + "=" * 60)
    print("PHASE 5 — EVALUATION")
    print("=" * 60)

    results = compute_all_metrics(trained, X_test, X_test_sc, y_test)
    # Attach y_test to results for charting
    for name in results:
        results[name]["_y_test"] = y_test

    plot_model_comparison(results, cv_results)
    plot_roc_curves_fixed(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_precision_recall(results, y_test)
    plot_feature_importance(results, feature_names)
    threshold_analysis(results, y_test)
    save_metrics(results, cv_results)
    print_recommendation(results, cv_results)

    print("\n[Phase 5 complete] All evaluation charts and metrics saved.")
    return results


if __name__ == "__main__":
    from phase3_data_preparation import prepare_data
    from phase4_modelling import run_modelling

    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc,
     feature_names, scaler, encoders) = prepare_data(save=False)

    trained, cv_results = run_modelling(
        X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    )

    run_evaluation(trained, X_test, X_test_sc, y_test, feature_names, cv_results)
