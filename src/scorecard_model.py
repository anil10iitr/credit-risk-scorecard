"""
Credit Scorecard Model Pipeline
================================
Champion/Challenger framework:
  - Champion: Logistic Regression scorecard (interpretable, regulator-preferred)
  - Challenger: XGBoost (performance benchmark)

Scoring follows FICO-style points scaling:
  Score = Offset + Factor * ln(odds)
  Where: PDO=20, Base Score=600, Base Odds=30:1
"""

import numpy as np
import pandas as pd
import yaml
import joblib
import os
from typing import Optional, Tuple, Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import logging

from src.woe_encoder import WOEBinEncoder
from src.evaluation import gini_coefficient, ks_statistic, psi, model_report, plot_model_report

logger = logging.getLogger(__name__)


# ==============================================================================
# Scorecard Scaling (FICO-style points)
# ==============================================================================

class ScorecardScaler:
    """
    Convert log-odds to points using standard PDO (Points to Double Odds) scaling.

    Formula:
      Factor = PDO / ln(2)
      Offset = Base_Score - Factor * ln(Base_Odds)
      Score  = Offset + Factor * ln(p/(1-p))

    Standard bank parameters:
      PDO = 20 (score drops 20 points when odds double)
      Base Score = 600
      Base Odds  = 30 (30 good:1 bad at score 600)
    """
    def __init__(self, pdo: int = 20, base_score: int = 600, base_odds: int = 30):
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def proba_to_score(self, prob_default: np.ndarray) -> np.ndarray:
        """Convert P(default) to scorecard points. Higher score = lower risk."""
        # Avoid log(0)
        prob_default = np.clip(prob_default, 1e-6, 1 - 1e-6)
        odds_good = (1 - prob_default) / prob_default
        score = self.offset + self.factor * np.log(odds_good)
        return np.round(score).astype(int)

    def score_to_proba(self, score: np.ndarray) -> np.ndarray:
        """Convert scorecard points back to P(default)."""
        log_odds = (score - self.offset) / self.factor
        odds = np.exp(log_odds)
        prob_good = odds / (1 + odds)
        return 1 - prob_good


# ==============================================================================
# Champion: Logistic Regression Scorecard
# ==============================================================================

class LogisticScorecard:
    """
    Logistic regression scorecard with WOE feature encoding.
    Preferred by regulators for interpretability.
    """

    def __init__(
        self,
        woe_encoder: WOEBinEncoder,
        scaler_params: Optional[Dict] = None,
        lr_params: Optional[Dict] = None,
    ):
        self.woe_encoder = woe_encoder
        self.scaler_params = scaler_params or {"pdo": 20, "base_score": 600, "base_odds": 30}
        self.lr_params = lr_params or {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42}

        self.model = LogisticRegression(**self.lr_params)
        self.scorecard_scaler = ScorecardScaler(**self.scaler_params)
        self.selected_features_: List[str] = []
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, iv_threshold: float = 0.02):
        """Fit WOE encoder → select features by IV → train logistic regression."""
        logger.info("Fitting WOE encoder...")
        X_woe = self.woe_encoder.fit_transform(X_train, y_train)

        # Select features by IV
        self.selected_features_ = self.woe_encoder.selected_features(threshold=iv_threshold)
        logger.info(f"  Selected {len(self.selected_features_)} features with IV ≥ {iv_threshold}")

        X_selected = X_woe[self.selected_features_].fillna(0)

        logger.info("Training Logistic Regression (champion model)...")
        self.model.fit(X_selected, y_train)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_woe = self.woe_encoder.transform(X)[self.selected_features_].fillna(0)
        return self.model.predict_proba(X_woe)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return FICO-style scorecard points (300–850 range)."""
        proba = self.predict_proba(X)
        return np.clip(self.scorecard_scaler.proba_to_score(proba), 300, 850)

    def get_scorecard_table(self) -> pd.DataFrame:
        """
        Generate full scorecard table: variable → bin → points.
        This is what gets submitted to MRM/regulators.
        """
        self._check_fitted()
        rows = []
        coefs = dict(zip(self.selected_features_, self.model.coef_[0]))
        intercept = self.model.intercept_[0]
        n_vars = len(self.selected_features_)

        # Distribute intercept equally across variables (standard practice)
        intercept_per_var = (
            self.scorecard_scaler.offset + self.scorecard_scaler.factor * intercept
        ) / n_vars

        for feature in self.selected_features_:
            bin_table = self.woe_encoder.get_bin_table(feature)
            coef = coefs[feature]
            iv = self.woe_encoder.iv_[feature]

            for _, row in bin_table.iterrows():
                points = (
                    intercept_per_var
                    - self.scorecard_scaler.factor * coef * row["woe"]
                )
                rows.append({
                    "variable": feature,
                    "bin": row["bin"],
                    "woe": round(row["woe"], 4),
                    "iv": round(iv, 4),
                    "event_rate_pct": round(row["event_rate"] * 100, 2),
                    "coefficient": round(coef, 4),
                    "points": round(points, 1),
                })

        return pd.DataFrame(rows)

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before prediction")


# ==============================================================================
# Challenger: XGBoost
# ==============================================================================

class XGBoostChallenger:
    """
    XGBoost challenger model with SHAP explanations.
    Used as performance benchmark against the logistic champion.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,  # adjust for class imbalance
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names_: List[str] = []
        self.explainer_ = None
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        self.feature_names_ = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        logger.info("Training XGBoost (challenger model)...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
        )

        # Build SHAP explainer
        logger.info("Building SHAP explainer...")
        self.explainer_ = shap.TreeExplainer(self.model)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names_])[:, 1]

    def shap_summary(self, X: pd.DataFrame, max_display: int = 15, save_path: Optional[str] = None):
        """Generate SHAP beeswarm summary plot."""
        import matplotlib.pyplot as plt
        shap_values = self.explainer_.shap_values(X[self.feature_names_])
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X[self.feature_names_], max_display=max_display, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return shap_values

    def feature_importance_df(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ==============================================================================
# Champion-Challenger Comparison
# ==============================================================================

def compare_champion_challenger(
    champion: LogisticScorecard,
    challenger: XGBoostChallenger,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    Side-by-side metric comparison for MRM documentation.
    """
    results = []
    for model, name in [(champion, "Champion (LogReg)"), (challenger, "Challenger (XGBoost)")]:
        y_pred_train = model.predict_proba(X_train)
        y_pred_val   = model.predict_proba(X_val)

        report = model_report(y_train, y_pred_train, y_pred_val, y_val, model_name=name)
        results.append(report)

    return pd.DataFrame(results).set_index("model")


# ==============================================================================
# Model Persistence
# ==============================================================================

def save_model(model, path: str, model_name: str = "model"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, f"{model_name}.pkl"))
    logger.info(f"Model saved: {os.path.join(path, model_name + '.pkl')}")


def load_model(path: str, model_name: str = "model"):
    return joblib.load(os.path.join(path, f"{model_name}.pkl"))
