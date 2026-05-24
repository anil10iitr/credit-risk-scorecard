"""
Data Loading & Preprocessing for IEEE-CIS Fraud Detection
==========================================================
Dataset: https://www.kaggle.com/c/ieee-fraud-detection/data
Files needed: train_transaction.csv, train_identity.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

TARGET = "isFraud"

# High-cardinality categoricals to label-encode
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "M1","M2","M3","M4","M5","M6","M7","M8","M9", "DeviceType", "DeviceInfo"]


def load_raw(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load transaction and (optionally) identity tables."""
    import os
    txn_path = f"{data_dir}/train_transaction.csv"
    idn_path = f"{data_dir}/train_identity.csv"

    transactions = pd.read_csv(txn_path)
    logger.info(f"Transactions: {transactions.shape} | Fraud rate: {transactions[TARGET].mean():.4%}")

    identity = None
    if os.path.exists(idn_path):
        identity = pd.read_csv(idn_path)
        logger.info(f"Identity: {identity.shape}")

    return transactions, identity


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode high-cardinality categoricals."""
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    return df


def time_aware_split(
    df: pd.DataFrame,
    val_pct: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-aware split: last val_pct% of transactions by TransactionDT.
    Critical for fraud models — future cannot leak into training.
    """
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * (1 - val_pct))

    X = df_sorted.drop(columns=[TARGET])
    y = df_sorted[TARGET]

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    logger.info(f"Train fraud rate: {y_train.mean():.4%} | Val fraud rate: {y_val.mean():.4%}")

    return X_train, X_val, y_train, y_val
