"""
train.py — Fraud Detection Model Training Pipeline
====================================================
Usage:
    python train.py
    python train.py --data data/raw/ --output outputs/
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")

from src.feature_engineering import build_features
from src.model import XGBoostFraudModel, LightGBMFraudModel, FraudEnsemble
from src.evaluation import fraud_metrics_at_threshold, plot_fraud_report, recall_at_fpr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET = "isFraud"
DROP_COLS = ["TransactionID", TARGET]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/models", exist_ok=True)

    # ── 1. Load Data ───────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1: Loading IEEE-CIS Data")
    logger.info("=" * 55)
    transactions = pd.read_csv(f"{args.data}/train_transaction.csv")
    identity     = pd.read_csv(f"{args.data}/train_identity.csv") if \
                   os.path.exists(f"{args.data}/train_identity.csv") else None

    logger.info(f"  Transactions: {len(transactions):,} rows | Fraud rate: {transactions[TARGET].mean():.3%}")

    # ── 2. Feature Engineering ────────────────────────────────────────────
    logger.info("\nSTEP 2: Feature Engineering")
    logger.info("=" * 55)
    df = build_features(transactions, identity=identity)

    # ── 3. Prepare features ───────────────────────────────────────────────
    FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

    # Handle categoricals (XGBoost needs numeric)
    for col in df[FEATURE_COLS].select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes

    X = df[FEATURE_COLS].fillna(-999)  # -999 = XGBoost handles internally
    y = df[TARGET]

    # Train/Val split (time-aware: last 20% as validation)
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    logger.info(f"  Train fraud rate: {y_train.mean():.3%} | Val fraud rate: {y_val.mean():.3%}")

    # ── 4. Train XGBoost ──────────────────────────────────────────────────
    logger.info("\nSTEP 4: Training XGBoost")
    logger.info("=" * 55)
    xgb_model = XGBoostFraudModel(
        params=config.get("xgboost"),
        fn_cost=config["costs"]["fn_cost"],
        fp_cost=config["costs"]["fp_cost"],
    )
    xgb_model.fit(X_train, y_train, X_val, y_val)

    # ── 5. Train LightGBM ─────────────────────────────────────────────────
    logger.info("\nSTEP 5: Training LightGBM")
    logger.info("=" * 55)
    lgb_model = LightGBMFraudModel(params=config.get("lightgbm"))
    lgb_model.fit(X_train, y_train, X_val, y_val)

    # ── 6. Ensemble Blend ─────────────────────────────────────────────────
    logger.info("\nSTEP 6: Ensemble Blending")
    logger.info("=" * 55)
    ensemble = FraudEnsemble(xgb_model, lgb_model, xgb_weight=0.6)
    ensemble.calibrate_threshold(X_val, y_val)

    # ── 7. Evaluation ─────────────────────────────────────────────────────
    logger.info("\nSTEP 7: Model Evaluation")
    logger.info("=" * 55)
    opt_thresh = ensemble.threshold_optimizer.optimal_threshold_
    y_ens_score_val = ensemble.predict_proba(X_val)

    metrics = fraud_metrics_at_threshold(y_val.values, y_ens_score_val, opt_thresh)
    logger.info("\n  Ensemble Model — Validation Metrics:")
    for k, v in metrics.items():
        logger.info(f"    {k}: {v}")

    recall_5fpr = recall_at_fpr(y_val.values, y_ens_score_val, 0.05)
    logger.info(f"\n  Recall @ 5% FPR: {recall_5fpr:.4f}  ← key business metric")

    # Save evaluation report
    plot_fraud_report(
        y_train.values, xgb_model.predict_proba(X_train),
        y_val.values,   y_ens_score_val,
        optimal_threshold=opt_thresh,
        model_name="XGBoost+LightGBM Ensemble",
        save_path=f"{args.output}/fraud_model_report.png",
    )

    # Feature importance
    fi = xgb_model.feature_importance()
    fi.to_csv(f"{args.output}/feature_importance.csv", index=False)
    logger.info(f"\n  Top 10 features:\n{fi.head(10).to_string(index=False)}")

    # SHAP summary
    xgb_model.shap_summary(
        X_val.sample(min(1000, len(X_val)), random_state=42),
        save_path=f"{args.output}/shap_summary.png"
    )

    # ── 8. Save Models ────────────────────────────────────────────────────
    import joblib
    joblib.dump(ensemble, f"{args.output}/models/fraud_ensemble.pkl")
    logger.info(f"\n✅ Ensemble saved → {args.output}/models/fraud_ensemble.pkl")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data",   default="data/raw")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    main(args)
