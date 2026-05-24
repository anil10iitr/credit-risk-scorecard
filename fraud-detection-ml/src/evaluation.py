"""
Fraud Detection Evaluation Metrics
====================================
Fraud models are evaluated differently from standard classifiers.
AUC alone is misleading — a model can have AUC=0.95 and terrible recall.

Key metrics for fraud:
  - AUC-ROC: Discriminatory power across all thresholds
  - AUC-PR (Average Precision): Better for imbalanced classes
  - Recall @ fixed FPR: Catch rate at acceptable false alarm rate
  - Precision-Recall tradeoff: Business drives where on this curve we operate
  - Cost matrix savings: $ saved vs. baseline (no model / rule-only)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


def fraud_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    fn_cost: float = 250,
    fp_cost: float = 5,
) -> Dict:
    """Full metric suite at a given classification threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    total_fraud = y_true.sum()

    recall      = tp / (tp + fn + 1e-10)
    precision   = tp / (tp + fp + 1e-10)
    fpr         = fp / (fp + tn + 1e-10)
    f1          = 2 * precision * recall / (precision + recall + 1e-10)
    total_cost  = fn * fn_cost + fp * fp_cost

    # Savings vs no-model baseline (all transactions flagged at 100% FPR)
    baseline_fp_cost = (total - total_fraud) * fp_cost
    baseline_fn_cost = 0
    baseline_cost    = baseline_fp_cost + baseline_fn_cost
    cost_savings     = baseline_cost - total_cost

    return {
        "threshold": round(threshold, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
        "auc_pr": round(average_precision_score(y_true, y_score), 4),
        "total_cost": round(total_cost, 0),
        "cost_savings_vs_baseline": round(cost_savings, 0),
        "fraud_catch_rate_pct": round(recall * 100, 1),
        "false_alarm_rate_pct": round(fpr * 100, 2),
    }


def recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """
    What is the recall (fraud catch rate) when we allow
    target_fpr% false alarm rate?

    Standard industry question: "At 5% false alarm rate, how many frauds do we catch?"
    """
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    # Find closest FPR to target
    idx = np.argmin(np.abs(fpr_arr - target_fpr))
    return float(tpr_arr[idx])


def model_comparison_table(
    y_true: np.ndarray,
    models: Dict[str, np.ndarray],  # {model_name: y_score}
    threshold: float = 0.5,
    fn_cost: float = 250,
    fp_cost: float = 5,
) -> pd.DataFrame:
    """Compare multiple models side-by-side."""
    rows = []
    for name, y_score in models.items():
        metrics = fraud_metrics_at_threshold(y_true, y_score, threshold, fn_cost, fp_cost)
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model")


def plot_fraud_report(
    y_true_train, y_score_train,
    y_true_val,   y_score_val,
    optimal_threshold: float = 0.5,
    model_name: str = "XGBoost Fraud Model",
    save_path: Optional[str] = None,
):
    """
    4-panel fraud model evaluation report:
      [1] ROC Curve with recall@5%FPR annotation
      [2] Precision-Recall Curve with business operating point
      [3] Score Distribution by Class (fraud vs legit)
      [4] Cost Curve — total cost vs threshold
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Fraud Model Report — {model_name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel 1: ROC ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for y_t, y_s, label, c in [
        (y_true_train, y_score_train, "Train", "#2563EB"),
        (y_true_val,   y_score_val,   "Val",   "#DC2626"),
    ]:
        fpr, tpr, _ = roc_curve(y_t, y_s)
        auc = roc_auc_score(y_t, y_s)
        ax1.plot(fpr, tpr, color=c, lw=2, label=f"{label} AUC={auc:.3f}")
    # Mark 5% FPR
    recall_5fpr = recall_at_fpr(y_true_val, y_score_val, 0.05)
    ax1.scatter([0.05], [recall_5fpr], color="green", zorder=5, s=80,
                label=f"Recall@5%FPR={recall_5fpr:.2f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Precision-Recall ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, thresh_pr = precision_recall_curve(y_true_val, y_score_val)
    ap = average_precision_score(y_true_val, y_score_val)
    ax2.plot(rec, prec, color="#7C3AED", lw=2, label=f"AP={ap:.3f}")
    ax2.axhline(0.10, color="orange", linestyle="--", lw=1.5, label="10% Precision floor")
    ax2.set_xlabel("Recall (Fraud Catch Rate)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Score Distributions ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    legit_scores  = y_score_val[y_true_val == 0]
    fraud_scores  = y_score_val[y_true_val == 1]
    ax3.hist(legit_scores, bins=50, alpha=0.6, color="#2563EB", label="Legitimate", density=True)
    ax3.hist(fraud_scores, bins=50, alpha=0.6, color="#DC2626", label="Fraud", density=True)
    ax3.axvline(optimal_threshold, color="green", linestyle="--", lw=2,
                label=f"Threshold={optimal_threshold:.3f}")
    ax3.set_xlabel("Predicted Fraud Probability")
    ax3.set_ylabel("Density")
    ax3.set_title("Score Distribution by Class")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Cost Curve ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    recalls_curve = []
    for t in thresholds:
        m = fraud_metrics_at_threshold(y_true_val, y_score_val, t)
        costs.append(m["total_cost"])
        recalls_curve.append(m["recall"])

    ax4_twin = ax4.twinx()
    ax4.plot(thresholds, costs, color="#DC2626", lw=2, label="Total Cost ($)")
    ax4_twin.plot(thresholds, recalls_curve, color="#2563EB", lw=1.5,
                  linestyle="--", label="Recall")
    ax4.axvline(optimal_threshold, color="green", linestyle="--", lw=2, label="Optimal threshold")
    ax4.set_xlabel("Classification Threshold")
    ax4.set_ylabel("Total Business Cost ($)", color="#DC2626")
    ax4_twin.set_ylabel("Recall", color="#2563EB")
    ax4.set_title("Cost Curve — Threshold Optimization")
    ax4.grid(True, alpha=0.3)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Report saved to {save_path}")
    return fig
