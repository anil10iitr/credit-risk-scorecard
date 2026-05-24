"""
WOE (Weight of Evidence) Encoder with IV Calculation
======================================================
Production-grade implementation used in bank credit scoring pipelines.

Weight of Evidence transforms categorical and continuous features into
a single numerical scale that directly measures their relationship with
the binary target (default/no-default). It is the standard feature
engineering approach in regulatory-compliant scorecards.

WOE(i) = ln(Distribution of Events_i / Distribution of Non-Events_i)
IV(feature) = sum[(% Events - % Non-Events) * WOE]

IV Interpretation:
  < 0.02  → Useless predictor
  0.02–0.1 → Weak predictor
  0.1–0.3  → Medium predictor
  > 0.3    → Strong predictor (verify for overfitting)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import warnings
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Small constant to avoid log(0)
EPSILON = 1e-6


class WOEBinEncoder:
    """
    End-to-end WOE binning and encoding for credit scorecard development.

    Supports:
    - Automatic optimal binning for continuous variables (equal-frequency)
    - Monotonicity enforcement (required for regulatory scorecards)
    - Custom bin overrides for expert judgement overlays
    - Information Value (IV) computation per feature
    - Fit on train, transform train/validation/OOT consistently

    Parameters
    ----------
    n_bins : int
        Initial number of bins before monotonicity merging (default: 10)
    min_bin_pct : float
        Minimum % of total population per bin (default: 0.05 = 5%)
    enforce_monotone : bool
        If True, adjacent bins are merged until WOE is monotone (default: True)
    handle_missing : str
        How to handle NaN: 'separate_bin' creates a dedicated bin (default)
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_bin_pct: float = 0.05,
        enforce_monotone: bool = True,
        handle_missing: str = "separate_bin",
    ):
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.enforce_monotone = enforce_monotone
        self.handle_missing = handle_missing

        # Fitted state
        self.bin_maps_: Dict[str, pd.DataFrame] = {}   # feature → bin table
        self.iv_: Dict[str, float] = {}                # feature → IV score
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEBinEncoder":
        """Fit WOE bins on training data."""
        assert y.nunique() == 2, "Target must be binary (0/1)"
        self.feature_names_ = list(X.columns)

        for col in self.feature_names_:
            logger.info(f"  Binning: {col}")
            bin_table = self._fit_single_feature(X[col], y)
            self.bin_maps_[col] = bin_table
            self.iv_[col] = bin_table["iv_contribution"].sum()

        self.is_fitted_ = True
        logger.info(f"WOE fitting complete. {len(self.feature_names_)} features processed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace raw feature values with WOE values."""
        self._check_fitted()
        X_woe = X.copy()
        for col in self.feature_names_:
            X_woe[col] = self._transform_single_feature(X[col], col)
        return X_woe

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_iv_summary(self, threshold: float = 0.02) -> pd.DataFrame:
        """
        Return IV summary table sorted by predictive power.
        Features below threshold are flagged for exclusion.
        """
        self._check_fitted()
        df = pd.DataFrame(
            {"feature": list(self.iv_.keys()), "iv": list(self.iv_.values())}
        ).sort_values("iv", ascending=False).reset_index(drop=True)

        df["strength"] = pd.cut(
            df["iv"],
            bins=[-np.inf, 0.02, 0.1, 0.3, np.inf],
            labels=["Useless", "Weak", "Medium", "Strong"],
        )
        df["include"] = df["iv"] >= threshold
        return df

    def get_bin_table(self, feature: str) -> pd.DataFrame:
        """Return detailed bin statistics for a single feature."""
        self._check_fitted()
        if feature not in self.bin_maps_:
            raise KeyError(f"Feature '{feature}' not found. Available: {self.feature_names_}")
        return self.bin_maps_[feature]

    def selected_features(self, threshold: float = 0.02) -> List[str]:
        """Return list of features with IV above threshold."""
        return self.get_iv_summary(threshold).query("include == True")["feature"].tolist()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fit_single_feature(self, series: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Compute WOE bin table for one feature."""
        total_events = y.sum()
        total_non_events = (1 - y).sum()

        # Step 1: Initial binning
        bins = self._create_initial_bins(series)

        # Step 2: Build raw bin table
        bin_table = self._compute_bin_stats(series, y, bins, total_events, total_non_events)

        # Step 3: Merge small bins
        bin_table = self._merge_small_bins(bin_table, series, y, total_events, total_non_events)

        # Step 4: Enforce monotonicity
        if self.enforce_monotone:
            bin_table = self._enforce_monotonicity(bin_table, series, y, total_events, total_non_events)

        # Step 5: Handle missing values
        if self.handle_missing == "separate_bin" and series.isna().any():
            missing_table = self._compute_missing_bin(series, y, total_events, total_non_events)
            bin_table = pd.concat([bin_table, missing_table], ignore_index=True)

        # Step 6: Compute final WOE and IV
        bin_table = self._compute_woe_iv(bin_table, total_events, total_non_events)
        return bin_table

    def _create_initial_bins(self, series: pd.Series) -> np.ndarray:
        """Create equal-frequency bins for continuous features."""
        clean = series.dropna()
        if clean.nunique() <= self.n_bins:
            # Categorical-like: use unique values as bins
            return np.sort(clean.unique())
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.unique(np.percentile(clean, quantiles))
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _compute_bin_stats(
        self,
        series: pd.Series,
        y: pd.Series,
        bins: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> pd.DataFrame:
        """Assign observations to bins and compute event rates."""
        df = pd.DataFrame({"x": series, "y": y})
        df = df[df["x"].notna()].copy()

        if len(bins) > 2:
            df["bin"] = pd.cut(df["x"], bins=bins, include_lowest=True)
        else:
            df["bin"] = df["x"].astype(str)

        agg = (
            df.groupby("bin", observed=True)
            .agg(count=("y", "count"), events=("y", "sum"))
            .reset_index()
        )
        agg["non_events"] = agg["count"] - agg["events"]
        agg["bin_lower"] = bins[:-1] if len(bins) > 2 else agg["bin"]
        agg["bin_upper"] = bins[1:] if len(bins) > 2 else agg["bin"]
        return agg

    def _merge_small_bins(
        self, bin_table: pd.DataFrame, series: pd.Series, y: pd.Series,
        total_events: float, total_non_events: float
    ) -> pd.DataFrame:
        """Merge bins that fall below min_bin_pct threshold."""
        total = bin_table["count"].sum()
        threshold = self.min_bin_pct * total

        while True:
            small = bin_table[bin_table["count"] < threshold]
            if small.empty:
                break
            idx = small.index[0]
            # Merge with adjacent bin (prefer next, fall back to previous)
            if idx < len(bin_table) - 1:
                merge_idx = idx + 1
            else:
                merge_idx = idx - 1

            bin_table = self._merge_two_rows(bin_table, min(idx, merge_idx))
        return bin_table.reset_index(drop=True)

    def _enforce_monotonicity(
        self, bin_table: pd.DataFrame, series: pd.Series, y: pd.Series,
        total_events: float, total_non_events: float
    ) -> pd.DataFrame:
        """Merge adjacent bins until WOE trend is monotone."""
        bt = self._compute_woe_iv(bin_table.copy(), total_events, total_non_events)
        max_iterations = 50

        for _ in range(max_iterations):
            woe_vals = bt["woe"].values
            violations = []
            for i in range(len(woe_vals) - 1):
                if i > 0:
                    # Check if current direction flips
                    dir_prev = np.sign(woe_vals[i] - woe_vals[i - 1])
                    dir_curr = np.sign(woe_vals[i + 1] - woe_vals[i])
                    if dir_prev != 0 and dir_curr != 0 and dir_prev != dir_curr:
                        violations.append(i)

            if not violations:
                break

            # Merge at first violation
            merge_at = violations[0]
            bt = self._merge_two_rows(bt, merge_at)
            bt = self._compute_woe_iv(bt, total_events, total_non_events)

        return bt

    def _merge_two_rows(self, bin_table: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Merge row idx and idx+1 into a single bin."""
        bt = bin_table.copy()
        row_a = bt.iloc[idx]
        row_b = bt.iloc[idx + 1]

        merged = {
            "count": row_a["count"] + row_b["count"],
            "events": row_a["events"] + row_b["events"],
            "non_events": row_a["non_events"] + row_b["non_events"],
            "bin_lower": row_a.get("bin_lower", row_a.get("bin")),
            "bin_upper": row_b.get("bin_upper", row_b.get("bin")),
            "bin": f"{row_a.get('bin_lower', '')} to {row_b.get('bin_upper', '')}",
        }

        bt = bt.drop([bt.index[idx], bt.index[idx + 1]])
        new_row = pd.DataFrame([merged])
        bt = pd.concat([bt.iloc[:idx], new_row, bt.iloc[idx:]], ignore_index=True)
        return bt

    def _compute_missing_bin(
        self, series: pd.Series, y: pd.Series,
        total_events: float, total_non_events: float
    ) -> pd.DataFrame:
        """Compute stats for missing values as a separate bin."""
        mask = series.isna()
        n_missing = mask.sum()
        n_events = y[mask].sum()
        return pd.DataFrame([{
            "bin": "MISSING",
            "bin_lower": np.nan,
            "bin_upper": np.nan,
            "count": n_missing,
            "events": n_events,
            "non_events": n_missing - n_events,
        }])

    def _compute_woe_iv(
        self, bin_table: pd.DataFrame, total_events: float, total_non_events: float
    ) -> pd.DataFrame:
        """Compute WOE and IV contribution for each bin."""
        bt = bin_table.copy()
        bt["pct_events"] = (bt["events"] + EPSILON) / (total_events + EPSILON)
        bt["pct_non_events"] = (bt["non_events"] + EPSILON) / (total_non_events + EPSILON)
        bt["woe"] = np.log(bt["pct_events"] / bt["pct_non_events"])
        bt["iv_contribution"] = (bt["pct_events"] - bt["pct_non_events"]) * bt["woe"]
        bt["event_rate"] = bt["events"] / bt["count"]
        return bt

    def _transform_single_feature(self, series: pd.Series, feature: str) -> pd.Series:
        """Map raw values to WOE values using fitted bin table."""
        bin_table = self.bin_maps_[feature]
        result = pd.Series(index=series.index, dtype=float)

        for _, row in bin_table.iterrows():
            if row["bin"] == "MISSING":
                mask = series.isna()
            else:
                lower = row.get("bin_lower", -np.inf)
                upper = row.get("bin_upper", np.inf)
                if pd.isna(lower):
                    lower = -np.inf
                if pd.isna(upper):
                    upper = np.inf
                mask = (series >= lower) & (series < upper) & series.notna()
            result[mask] = row["woe"]

        # Fill any unmapped values with 0
        result = result.fillna(0.0)
        return result

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before .transform()")
