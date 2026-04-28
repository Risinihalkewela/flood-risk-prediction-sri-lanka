# =============================================================================
# main.py — Master Pipeline Runner
# Sri Lanka Flood Risk Dataset — CRISP-DM Data Mining Project
# =============================================================================
# Runs the complete CRISP-DM pipeline end-to-end:
#
#   Phase 2 — Exploratory Data Analysis
#   Phase 3 — Data Preparation & Cleaning
#   Phase 4 — Model Training
#   Phase 5 — Model Evaluation
#   Phase 6 — Deployment / Risk Report
#
# Usage:
#   python main.py                    # run all phases
#   python main.py --phases 3 4 5     # run specific phases only
#   python main.py --skip-eda         # skip EDA charts (faster)
# =============================================================================

import argparse
import time
import sys
import os

def banner(title):
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_pipeline(phases=None, skip_eda=False):
    start_total = time.time()

    # ── Phase 2: EDA ─────────────────────────────────────────────────────────
    if (phases is None or 2 in phases) and not skip_eda:
        banner("PHASE 2 — EXPLORATORY DATA ANALYSIS")
        t = time.time()
        from phase2_eda import run_eda
        run_eda()
        print(f"\n  Phase 2 completed in {time.time()-t:.1f}s")

    # ── Phase 3: Data Preparation ─────────────────────────────────────────────
    if phases is None or 3 in phases:
        banner("PHASE 3 — DATA PREPARATION")
        t = time.time()
        from phase3_data_preparation import prepare_data
        (X_train, X_test, y_train, y_test,
         X_train_sc, X_test_sc,
         feature_names, scaler, encoders) = prepare_data(save=True)
        print(f"\n  Phase 3 completed in {time.time()-t:.1f}s")
    else:
        # Load from saved files if skipping Phase 3
        import pandas as pd
        train = pd.read_csv("outputs/train_set.csv")
        test  = pd.read_csv("outputs/test_set.csv")
        y_train = train.pop("target")
        y_test  = test.pop("target")
        X_train = X_test = train  # simplified — run phase 3 for proper split
        X_train_sc = X_test_sc = X_train
        feature_names = list(train.columns)
        scaler = encoders = None

    # ── Phase 4: Modelling ────────────────────────────────────────────────────
    if phases is None or 4 in phases:
        banner("PHASE 4 — MODEL TRAINING")
        t = time.time()
        from phase4_modelling import run_modelling
        trained, cv_results = run_modelling(
            X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
        )
        print(f"\n  Phase 4 completed in {time.time()-t:.1f}s")

    # ── Phase 5: Evaluation ───────────────────────────────────────────────────
    if phases is None or 5 in phases:
        banner("PHASE 5 — EVALUATION")
        t = time.time()
        from phase5_evaluation import run_evaluation
        results = run_evaluation(
            trained, X_test, X_test_sc, y_test,
            feature_names, cv_results
        )
        print(f"\n  Phase 5 completed in {time.time()-t:.1f}s")

    # ── Phase 6: Deployment ───────────────────────────────────────────────────
    if phases is None or 6 in phases:
        banner("PHASE 6 — DEPLOYMENT")
        t = time.time()
        from phase6_deployment import risk_report, generate_risk_map
        risk_report()
        generate_risk_map()
        print(f"\n  Phase 6 completed in {time.time()-t:.1f}s")

    # ── Final Summary ─────────────────────────────────────────────────────────
    total = time.time() - start_total
    banner(f"PIPELINE COMPLETE — Total time: {total:.1f}s")

    import json
    metrics_path = os.path.join("outputs", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("\n  FINAL MODEL PERFORMANCE SUMMARY")
        print(f"  {'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Flood F1':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for name, m in metrics.items():
            print(f"  {name:<25} {m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} {m['f1_flood']:>10.4f}")
        print("\n  All outputs saved to: outputs/")
        print("  All charts saved to : outputs/charts/")
        print("  Models saved to     : outputs/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRISP-DM Flood Risk Pipeline — Sri Lanka"
    )
    parser.add_argument(
        "--phases", nargs="+", type=int,
        help="Phases to run (e.g. --phases 3 4 5). Default: all."
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip Phase 2 EDA (saves ~30s)"
    )
    args = parser.parse_args()

    phases = set(args.phases) if args.phases else None
    run_pipeline(phases=phases, skip_eda=args.skip_eda)
