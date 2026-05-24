"""
predict.py — Score New Loan Applicants
=======================================
Usage:
    python predict.py --input data/new_applicants.csv --output outputs/scores.csv
"""

import argparse
import logging
import pandas as pd
import numpy as np
import joblib

from src.data_utils import preprocess, FEATURE_COLS
from src.scorecard_model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    # Load models
    logger.info("Loading models...")
    champion   = load_model("outputs/models", "champion_logistic")
    woe_encoder = load_model("outputs/models", "woe_encoder")

    # Load applicant data
    logger.info(f"Loading applicant data: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"  {len(df):,} applicants to score")

    # Preprocess
    df_clean = preprocess(df, cap_outliers=True, impute_missing=True)
    X = df_clean[FEATURE_COLS]

    # Generate scores and probabilities
    proba   = champion.predict_proba(X)
    scores  = champion.predict_score(X)

    # Assign decisions
    decisions = pd.cut(
        scores,
        bins=[0, 580, 620, 850],
        labels=["DECLINE", "REVIEW", "APPROVE"],
    )

    # Build output
    output = df.copy()
    output["predicted_default_prob"] = proba.round(4)
    output["scorecard_points"]       = scores
    output["decision"]               = decisions

    output.to_csv(args.output, index=False)
    logger.info(f"\nScoring complete. Results saved to {args.output}")

    # Summary
    logger.info("\nDecision Distribution:")
    logger.info(decisions.value_counts().to_string())
    logger.info(f"\nAverage Score: {scores.mean():.1f}")
    logger.info(f"Score Range:   {scores.min()} – {scores.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="CSV with applicant features")
    parser.add_argument("--output", default="outputs/scores.csv")
    args = parser.parse_args()
    main(args)
