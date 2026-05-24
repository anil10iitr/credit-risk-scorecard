"""
train.py — Full Credit Scorecard Training Pipeline
====================================================
Runs end-to-end: data load → preprocessing → WOE → train → evaluate → save.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --data data/raw/cs-training.csv --output outputs/
"""

import argparse
import logging
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments

from src.data_utils import load_data, preprocess, split_data, data_quality_report, FEATURE_COLS
from src.woe_encoder import WOEBinEncoder
from src.scorecard_model import (
    LogisticScorecard,
    XGBoostChallenger,
    compare_champion_challenger,
    save_model,
)
from src.evaluation import plot_model_report, decile_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # ─── 1. Load & Validate Data ───────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1: Data Loading & Quality Check")
    logger.info("=" * 55)
    df = load_data(args.data)
    quality_report = data_quality_report(df)
    quality_report.to_csv(f"{args.output}/data_quality_report.csv", index=False)
    logger.info(f"  Data quality report saved → {args.output}/data_quality_report.csv")

    # ─── 2. Preprocessing ──────────────────────────────────────────────────
    logger.info("\nSTEP 2: Preprocessing")
    logger.info("=" * 55)
    df_clean = preprocess(df, cap_outliers=True, impute_missing=True)

    # ─── 3. Train / Val / OOT Split ────────────────────────────────────────
    logger.info("\nSTEP 3: Data Split")
    logger.info("=" * 55)
    X_train, X_val, X_oot, y_train, y_val, y_oot = split_data(
        df_clean,
        val_size=config["split"]["val_size"],
        oot_size=config["split"]["oot_size"],
        random_state=config["split"]["random_state"],
    )

    # ─── 4. WOE Encoding & IV Selection ────────────────────────────────────
    logger.info("\nSTEP 4: WOE Binning & Feature Selection")
    logger.info("=" * 55)
    woe_encoder = WOEBinEncoder(
        n_bins=config["woe"]["n_bins"],
        min_bin_pct=config["woe"]["min_bin_pct"],
        enforce_monotone=config["woe"]["enforce_monotone"],
    )

    # Champion model fits WOE internally
    champion = LogisticScorecard(
        woe_encoder=woe_encoder,
        scaler_params=config["scorecard_scaler"],
        lr_params=config["logistic_regression"],
    )
    champion.fit(X_train, y_train, iv_threshold=config["woe"]["iv_threshold"])

    # Save IV summary
    iv_summary = woe_encoder.get_iv_summary()
    iv_summary.to_csv(f"{args.output}/iv_summary.csv", index=False)
    logger.info(f"  IV summary saved → {args.output}/iv_summary.csv")
    logger.info("\n  Top features by Information Value:")
    logger.info(iv_summary.head(10).to_string(index=False))

    # Save full scorecard table
    scorecard_table = champion.get_scorecard_table()
    scorecard_table.to_csv(f"{args.output}/scorecard_points.csv", index=False)
    logger.info(f"\n  Scorecard points table saved → {args.output}/scorecard_points.csv")

    # ─── 5. Champion Evaluation ────────────────────────────────────────────
    logger.info("\nSTEP 5: Champion Model Evaluation (Logistic Scorecard)")
    logger.info("=" * 55)
    y_pred_train = champion.predict_proba(X_train)
    y_pred_val   = champion.predict_proba(X_val)
    y_pred_oot   = champion.predict_proba(X_oot)

    plot_model_report(
        y_train.values, y_pred_train,
        y_val.values, y_pred_val,
        model_name="Champion — Logistic Scorecard",
        save_path=f"{args.output}/champion_report.png",
    )

    # OOT evaluation
    from src.evaluation import gini_coefficient, ks_statistic, psi
    logger.info("\n  Out-of-Time (OOT) Performance:")
    logger.info(f"    Gini (OOT): {gini_coefficient(y_oot.values, y_pred_oot):.4f}")
    ks_oot, _ = ks_statistic(y_oot.values, y_pred_oot)
    logger.info(f"    KS   (OOT): {ks_oot:.4f}")
    logger.info(f"    PSI (train→OOT): {psi(y_pred_train, y_pred_oot):.4f}")

    # Decile table
    dec = decile_table(y_val.values, y_pred_val)
    dec.to_csv(f"{args.output}/decile_table.csv", index=False)
    logger.info(f"\n  Decile table saved → {args.output}/decile_table.csv")
    logger.info("\n  Score Deciles:")
    logger.info(dec[["decile", "event_rate", "cumulative_pct_events", "lift", "ks"]].to_string(index=False))

    # ─── 6. Challenger: XGBoost ────────────────────────────────────────────
    logger.info("\nSTEP 6: Challenger Model — XGBoost")
    logger.info("=" * 55)
    challenger = XGBoostChallenger(params=config["xgboost"])
    challenger.fit(X_train, y_train, X_val, y_val)

    plot_model_report(
        y_train.values, challenger.predict_proba(X_train),
        y_val.values,   challenger.predict_proba(X_val),
        model_name="Challenger — XGBoost",
        save_path=f"{args.output}/challenger_report.png",
    )

    # SHAP summary
    challenger.shap_summary(
        X_val.sample(min(500, len(X_val)), random_state=42),
        save_path=f"{args.output}/shap_summary.png"
    )

    # ─── 7. Champion vs Challenger ─────────────────────────────────────────
    logger.info("\nSTEP 7: Champion vs Challenger Comparison")
    logger.info("=" * 55)
    comparison = compare_champion_challenger(
        champion, challenger, X_train, y_train, X_val, y_val
    )
    comparison.to_csv(f"{args.output}/champion_challenger_comparison.csv")
    logger.info("\n" + comparison.to_string())

    # ─── 8. Save Models ────────────────────────────────────────────────────
    logger.info("\nSTEP 8: Saving Models")
    logger.info("=" * 55)
    save_model(champion,    "outputs/models", "champion_logistic")
    save_model(challenger,  "outputs/models", "challenger_xgboost")
    save_model(woe_encoder, "outputs/models", "woe_encoder")

    logger.info("\n✅ Pipeline complete. Check outputs/ for all reports and artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Scorecard Training Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data",   default="data/raw/cs-training.csv", help="Path to training data")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()
    main(args)
