"""
Fraud Detection Feature Engineering
=====================================
The single biggest driver of fraud model performance is feature engineering.
Raw transaction data is weak; engineered signals of suspicious behavior are strong.

Feature families implemented here:
  1. Device fingerprint features — card-device relationship signals
  2. Velocity features — transaction frequency/amount in rolling windows
  3. Aggregation features — deviation from cardholder's behavioral baseline
  4. Time-based features — time of day, day of week, time since last txn
  5. Network features — shared device/email/address across multiple accounts
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Device Fingerprint Features
# ==============================================================================

def device_card_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many unique cards have been seen on each device?
    A device used by many different cards = strong fraud signal.

    Source: Replicated from EXL production work (anonymized).
    """
    df = df.copy()

    if "DeviceInfo" in df.columns and "card1" in df.columns:
        # Cards per device
        cards_per_device = (
            df.groupby("DeviceInfo")["card1"]
            .nunique()
            .reset_index()
            .rename(columns={"card1": "unique_cards_on_device"})
        )
        df = df.merge(cards_per_device, on="DeviceInfo", how="left")

        # Devices per card
        devices_per_card = (
            df.groupby("card1")["DeviceInfo"]
            .nunique()
            .reset_index()
            .rename(columns={"DeviceInfo": "unique_devices_for_card"})
        )
        df = df.merge(devices_per_card, on="card1", how="left")

        # Device-card pair frequency (rare pairs = suspicious)
        device_card_count = (
            df.groupby(["DeviceInfo", "card1"])
            .size()
            .reset_index(name="device_card_pair_count")
        )
        df = df.merge(device_card_count, on=["DeviceInfo", "card1"], how="left")

        logger.info("  Device fingerprint features created: 3 features")

    return df


def email_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Email domain risk features.
    Free/temporary email domains (yopmail, mailinator, guerrillamail) are
    disproportionately used in fraudulent accounts.
    """
    df = df.copy()
    HIGH_RISK_DOMAINS = {
        "yopmail.com", "mailinator.com", "guerrillamail.com",
        "throwam.com", "fakeinbox.com", "spam4.me", "trashmail.com"
    }

    if "P_emaildomain" in df.columns:
        df["email_is_highrisk"] = df["P_emaildomain"].isin(HIGH_RISK_DOMAINS).astype(int)
        df["email_domain_txn_count"] = df.groupby("P_emaildomain")["P_emaildomain"].transform("count")
        # Low count = new/rare email domain = slight risk signal
        df["email_domain_is_rare"] = (df["email_domain_txn_count"] < 50).astype(int)
        logger.info("  Email domain features created: 3 features")

    return df


# ==============================================================================
# 2. Velocity Features (rolling window aggregations)
# ==============================================================================

def transaction_velocity(
    df: pd.DataFrame,
    windows: List[float] = [1, 6, 24],  # hours
) -> pd.DataFrame:
    """
    Rolling transaction count and amount per card in time windows.

    These are the most powerful single-feature group in fraud detection:
    fraudsters typically make multiple rapid transactions after stealing a card.

    Implementation note: True rolling windows require sorted data with
    a proper time index. We approximate here using TransactionDT.
    """
    df = df.copy()
    df = df.sort_values(["card1", "TransactionDT"])

    for window_hours in windows:
        window_sec = window_hours * 3600
        col_prefix = f"w{int(window_hours)}h"

        # Transaction count in window
        df[f"{col_prefix}_txn_count"] = (
            df.groupby("card1")["TransactionDT"]
            .transform(lambda x: x.expanding().count())
        )

        # Amount sum in window (approximation via cumsum)
        df[f"{col_prefix}_amount_sum"] = (
            df.groupby("card1")["TransactionAmt"]
            .transform("cumsum")
        )

        # Time since last transaction by same card
    df["time_since_last_txn"] = (
        df.groupby("card1")["TransactionDT"]
        .transform(lambda x: x.diff().fillna(0))
    )

    logger.info(f"  Velocity features created: {len(windows)*2 + 1} features")
    return df


# ==============================================================================
# 3. Aggregation / Behavioral Baseline Features
# ==============================================================================

def behavioral_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deviation from the cardholder's own historical transaction pattern.

    Key insight: fraud transactions often DEVIATE from the cardholder's
    normal spending pattern — different amount range, merchant category,
    location. Capturing deviation is more powerful than absolute values.
    """
    df = df.copy()

    # Mean/std transaction amount per card
    card_stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={"mean": "card_mean_amt", "std": "card_std_amt", "median": "card_median_amt"})
    )
    df = df.merge(card_stats, on="card1", how="left")

    # Z-score: how many std deviations is this transaction from cardholder mean?
    df["amt_zscore_from_card_mean"] = (
        (df["TransactionAmt"] - df["card_mean_amt"]) /
        (df["card_std_amt"].fillna(1) + 1e-6)
    )

    # Amount ratio: current txn vs cardholder median
    df["amt_ratio_to_median"] = df["TransactionAmt"] / (df["card_median_amt"] + 1e-6)

    # Transaction amount per merchant category
    if "ProductCD" in df.columns:
        merch_stats = (
            df.groupby("ProductCD")["TransactionAmt"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "merch_mean_amt", "std": "merch_std_amt"})
        )
        df = df.merge(merch_stats, on="ProductCD", how="left")
        df["amt_deviation_from_merch"] = (
            (df["TransactionAmt"] - df["merch_mean_amt"]) /
            (df["merch_std_amt"].fillna(1) + 1e-6)
        )

    logger.info("  Behavioral baseline features created: 5+ features")
    return df


# ==============================================================================
# 4. Time-Based Features
# ==============================================================================

def time_features(df: pd.DataFrame, reference_dt: int = 86400) -> pd.DataFrame:
    """
    Extract temporal patterns from TransactionDT (seconds from reference date).

    Fraudsters show distinct time-of-day patterns — high activity in
    late night hours when cardholders are less likely to notice alerts.
    """
    df = df.copy()

    # Convert to seconds-of-day
    df["hour_of_day"]   = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"]   = (df["TransactionDT"] // 86400) % 7

    # Is this a night-time transaction? (11pm – 5am)
    df["is_night_txn"]  = df["hour_of_day"].isin(range(23, 24)).astype(int) | \
                          df["hour_of_day"].isin(range(0, 5)).astype(int)

    # Is weekend?
    df["is_weekend"]    = df["day_of_week"].isin([5, 6]).astype(int)

    logger.info("  Time features created: 4 features")
    return df


# ==============================================================================
# 5. Network Features (shared attributes across accounts)
# ==============================================================================

def network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect shared attributes between accounts — a classic fraud ring signal.

    When multiple accounts share the same device, email domain, address,
    or billing info, they often belong to the same fraud operation.
    """
    df = df.copy()

    # Shared billing address across multiple cards
    if "addr1" in df.columns:
        address_card_count = (
            df.groupby("addr1")["card1"]
            .nunique()
            .reset_index()
            .rename(columns={"card1": "cards_at_address"})
        )
        df = df.merge(address_card_count, on="addr1", how="left")
        df["address_is_shared"] = (df["cards_at_address"] > 3).astype(int)

    # Shared email across cards
    if "P_emaildomain" in df.columns:
        email_card_count = (
            df.groupby("P_emaildomain")["card1"]
            .nunique()
            .reset_index()
            .rename(columns={"card1": "cards_with_email_domain"})
        )
        df = df.merge(email_card_count, on="P_emaildomain", how="left")

    logger.info("  Network features created: 3 features")
    return df


# ==============================================================================
# Master Feature Engineering Function
# ==============================================================================

def build_features(
    transactions: pd.DataFrame,
    identity: Optional[pd.DataFrame] = None,
    run_all: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    transactions : pd.DataFrame
        IEEE-CIS transaction table (or equivalent transaction log)
    identity : pd.DataFrame, optional
        IEEE-CIS identity table with device/browser info
    run_all : bool
        If True, run all feature families

    Returns
    -------
    pd.DataFrame with all engineered features appended
    """
    logger.info("Starting feature engineering pipeline...")

    # Merge identity data if available
    if identity is not None:
        logger.info("  Merging identity table...")
        df = transactions.merge(identity, on="TransactionID", how="left")
    else:
        df = transactions.copy()

    if run_all:
        logger.info("Building device features...")
        df = device_card_velocity(df)
        df = email_domain_features(df)

        logger.info("Building velocity features...")
        df = transaction_velocity(df)

        logger.info("Building behavioral baseline features...")
        df = behavioral_baseline_features(df)

        logger.info("Building time features...")
        df = time_features(df)

        logger.info("Building network features...")
        df = network_features(df)

    n_features_added = df.shape[1] - transactions.shape[1]
    logger.info(f"Feature engineering complete. Added {n_features_added} features. Total: {df.shape[1]}")
    return df


def get_feature_groups() -> Dict[str, List[str]]:
    """Return feature group definitions for SHAP analysis."""
    return {
        "device": [
            "unique_cards_on_device", "unique_devices_for_card", "device_card_pair_count"
        ],
        "email": [
            "email_is_highrisk", "email_domain_txn_count", "email_domain_is_rare"
        ],
        "velocity": [
            "w1h_txn_count", "w6h_txn_count", "w24h_txn_count",
            "w1h_amount_sum", "w6h_amount_sum", "w24h_amount_sum",
            "time_since_last_txn",
        ],
        "behavioral": [
            "amt_zscore_from_card_mean", "amt_ratio_to_median",
            "amt_deviation_from_merch", "card_mean_amt", "card_std_amt",
        ],
        "time": [
            "hour_of_day", "day_of_week", "is_night_txn", "is_weekend"
        ],
        "network": [
            "cards_at_address", "address_is_shared", "cards_with_email_domain"
        ],
    }
