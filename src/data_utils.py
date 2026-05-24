"""
Data Loading & Preprocessing Utilities
=======================================
Handles the Give Me Some Credit (Kaggle) dataset.
Download: https://www.kaggle.com/c/GiveMeSomeCredit/data
Place cs-training.csv in data/raw/
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Dataset schema
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

TARGET_COL = "SeriousDlqin2yrs"

# Business-driven caps to handle extreme outliers
OUTLIER_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": (0, 1.5),
    "age": (18, 100),
    "NumberOfTime30-59DaysPastDueNotWorse": (0, 20),
    "DebtRatio": (0, 10),
    "MonthlyIncome": (0, 50_000),
    "NumberOfOpenCreditLinesAndLoans": (0, 40),
    "NumberOfTimes90DaysLate": (0, 20),
    "NumberRealEstateLoansOrLines": (0, 10),
    "NumberOfTime60-89DaysPastDueNotWorse": (0, 20),
    "NumberOfDependents": (0, 10),
}


def load_data(path: str = "data/raw/cs-training.csv") -> pd.DataFrame:
    """Load raw dataset with validation."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, index_col=0)
    logger.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    logger.info(f"  Target rate: {df[TARGET_COL].mean():.2%}")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data quality summary table."""
    report = pd.DataFrame({
        "feature": df.columns,
        "dtype": df.dtypes.values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
        "unique_values": df.nunique().values,
        "min": df.min().values,
        "max": df.max().values,
        "mean": df.mean().round(4).values,
        "median": df.median().round(4).values,
    })
    return report


def preprocess(
    df: pd.DataFrame,
    cap_outliers: bool = True,
    impute_missing: bool = True,
) -> pd.DataFrame:
    """
    Apply business-driven preprocessing:
    1. Cap extreme outliers using domain knowledge
    2. Impute missing values (median imputation — conservative for MRM)
    """
    df = df.copy()

    # Cap outliers
    if cap_outliers:
        for col, (low, high) in OUTLIER_CAPS.items():
            if col in df.columns:
                original_outliers = ((df[col] < low) | (df[col] > high)).sum()
                df[col] = df[col].clip(lower=low, upper=high)
                if original_outliers > 0:
                    logger.info(f"  Capped {original_outliers} outliers in {col}")

    # Impute missing with median
    if impute_missing:
        for col in FEATURE_COLS:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                n_missing = df[col].isna().sum()
                df[col] = df[col].fillna(median_val)
                logger.info(f"  Imputed {n_missing} missing values in {col} with median={median_val:.2f}")

    return df


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    oot_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,  pd.Series,  pd.Series]:
    """
    Three-way split: Train / Validation / Out-of-Time (OOT).
    OOT simulates model performance on future data — key for MRM.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # First split off OOT
    X_dev, X_oot, y_dev, y_oot = train_test_split(
        X, y, test_size=oot_size, random_state=random_state, stratify=y
    )

    # Then split dev into train/val
    adjusted_val = val_size / (1 - oot_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=adjusted_val, random_state=random_state, stratify=y_dev
    )

    logger.info(f"  Train: {len(X_train):,} rows ({y_train.mean():.2%} bad rate)")
    logger.info(f"  Val:   {len(X_val):,} rows  ({y_val.mean():.2%} bad rate)")
    logger.info(f"  OOT:   {len(X_oot):,} rows  ({y_oot.mean():.2%} bad rate)")

    return X_train, X_val, X_oot, y_train, y_val, y_oot
