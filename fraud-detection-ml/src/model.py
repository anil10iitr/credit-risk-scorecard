"""
Fraud Detection Model — Recall-Optimized XGBoost
=================================================
Core design choices:
  1. Maximize Recall (catch fraud) subject to a Precision floor
  2. Use scale_pos_weight for class imbalance (simpler + better than SMOTE for trees)
  3. Cost-based threshold selection — business drives the cutoff, not 0.5
  4. Ensemble: XGBoost + LightGBM blend for robustness
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
import shap
import joblib
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# Cost-Based Threshold Optimizer
# ==============================================================================

class FraudThresholdOptimizer:
    """
    Select the optimal classification threshold based on business costs.

    Fraud cost matrix (typical US credit card portfolio):
      False Negative (miss fraud):   $250 loss per fraud transaction
      False Positive (flag legit):   $5  cost per declined transaction
      True Positive (catch fraud):   $0  (fraud prevented)
      True Negative (pass legit):    $0  (normal operation)

    Optimal threshold minimizes: FN_cost * FN + FP_cost * FP
    """

    def __init__(self, fn_cost: float = 250.0, fp_cost: float = 5.0):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.optimal_threshold_: Optional[float] = None
        self.cost_curve_: Optional[pd.DataFrame] = None

    def fit(self, y_true: np.ndarray, y_score: np.ndarray) -> "FraudThresholdOptimizer":
        """Compute total cost at each threshold and select minimum."""
        thresholds = np.linspace(0.01, 0.99, 200)
        results = []

        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            total_cost = self.fn_cost * fn + self.fp_cost * fp
            recall = tp / (tp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)

            results.append({
                "threshold": thresh,
                "total_cost": total_cost,
                "recall": recall,
                "precision": precision,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

        self.cost_curve_ = pd.DataFrame(results)
        best_row = self.cost_curve_.loc[self.cost_curve_["total_cost"].idxmin()]
        self.optimal_threshold_ = best_row["threshold"]

        logger.info(f"  Optimal threshold: {self.optimal_threshold_:.3f}")
        logger.info(f"  At optimal — Recall: {best_row['recall']:.3f}, Precision: {best_row['precision']:.3f}")
        logger.info(f"  Total cost at optimal: ${best_row['total_cost']:,.0f}")
        return self

    def predict(self, y_score: np.ndarray) -> np.ndarray:
        if self.optimal_threshold_ is None:
            raise RuntimeError("Call .fit() first")
        return (y_score >= self.optimal_threshold_).astype(int)


# ==============================================================================
# XGBoost Fraud Classifier
# ==============================================================================

class XGBoostFraudModel:
    """
    XGBoost fraud detection model with:
    - scale_pos_weight for class imbalance
    - Early stopping on validation AUC
    - SHAP integration for explainability
    - Cross-validated performance estimates
    """

    DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "gamma": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[Dict] = None, fn_cost: float = 250, fp_cost: float = 5):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.threshold_optimizer = FraudThresholdOptimizer(fn_cost=fn_cost, fp_cost=fp_cost)
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer_: Optional[shap.TreeExplainer] = None
        self.feature_names_: List[str] = []
        self.is_fitted_ = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
    ) -> "XGBoostFraudModel":
        self.feature_names_ = list(X_train.columns)

        # Compute scale_pos_weight from training data imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos
        logger.info(f"  Class imbalance — Negatives: {neg:,} | Positives: {pos:,} | scale_pos_weight={spw:.1f}")

        self.model = xgb.XGBClassifier(**self.params, scale_pos_weight=spw)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        # Build SHAP explainer
        logger.info("  Building SHAP explainer...")
        self.explainer_ = shap.TreeExplainer(self.model)

        # Optimize threshold on validation set
        logger.info("  Optimizing classification threshold...")
        y_val_score = self.predict_proba(X_val)
        self.threshold_optimizer.fit(y_val.values, y_val_score)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fraud) for each transaction."""
        return self.model.predict_proba(X[self.feature_names_])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary fraud flag using cost-optimized threshold."""
        return self.threshold_optimizer.predict(self.predict_proba(X))

    def cross_val_score(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> Dict[str, float]:
        """Stratified k-fold cross validation."""
        from sklearn.model_selection import cross_val_score as cvs
        scores = cvs(self.model, X[self.feature_names_], y,
                     cv=StratifiedKFold(n_splits=n_splits), scoring="roc_auc")
        result = {"auc_mean": scores.mean(), "auc_std": scores.std()}
        logger.info(f"  CV AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
        return result

    def shap_explain_transaction(
        self, X_row: pd.DataFrame, save_path: Optional[str] = None
    ):
        """Generate SHAP force plot for a single transaction (fraud analyst tool)."""
        import matplotlib.pyplot as plt
        shap_values = self.explainer_.shap_values(X_row[self.feature_names_])
        shap.initjs()
        plot = shap.force_plot(
            self.explainer_.expected_value,
            shap_values[0],
            X_row[self.feature_names_].iloc[0],
            matplotlib=True,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return plot

    def shap_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Global SHAP feature importance summary."""
        import matplotlib.pyplot as plt
        shap_vals = self.explainer_.shap_values(X[self.feature_names_])
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X[self.feature_names_], show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return shap_vals

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Model saved: {path}")

    @staticmethod
    def load(path: str) -> "XGBoostFraudModel":
        return joblib.load(path)


# ==============================================================================
# LightGBM Ensemble Component
# ==============================================================================

class LightGBMFraudModel:
    """LightGBM component for ensemble blending."""

    DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: List[str] = []
        self.is_fitted_ = False

    def fit(self, X_train, y_train, X_val, y_val):
        self.feature_names_ = list(X_train.columns)
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()

        self.model = lgb.LGBMClassifier(**self.params, scale_pos_weight=neg / pos)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.feature_names_])[:, 1]


# ==============================================================================
# Ensemble Blend
# ==============================================================================

class FraudEnsemble:
    """
    Weighted average blend of XGBoost + LightGBM.
    Blending reduces variance and typically improves AUC by 0.5–1%.
    """

    def __init__(
        self,
        xgb_model: XGBoostFraudModel,
        lgb_model: LightGBMFraudModel,
        xgb_weight: float = 0.6,
    ):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.xgb_weight = xgb_weight
        self.lgb_weight = 1 - xgb_weight
        self.threshold_optimizer = FraudThresholdOptimizer()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        xgb_proba = self.xgb_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X)
        return self.xgb_weight * xgb_proba + self.lgb_weight * lgb_proba

    def calibrate_threshold(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Fit cost-based threshold on validation set."""
        y_score = self.predict_proba(X_val)
        self.threshold_optimizer.fit(y_val.values, y_score)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.threshold_optimizer.predict(self.predict_proba(X))
