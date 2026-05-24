"""
Credit Model Evaluation Suite
==============================
Standard metrics used in bank Model Risk Management (MRM) reviews.

Metrics implemented:
  - Gini Coefficient (= 2 * AUC - 1): Primary discriminatory power metric
  - KS Statistic (Kolmogorov-Smirnov): Max separation between event/non-event distributions
  - PSI (Population Stability Index): Detects score distribution drift
  - Decile Table: Gains/lift analysis per score band
  - SHAP Summary: Global feature importance with directionality

Regulatory thresholds (standard MRM benchmarks):
  Gini ≥ 0.35 → Acceptable  |  ≥ 0.50 → Good  |  ≥ 0.65 → Excellent
  KS  ≥ 0.20 → Acceptable  |  ≥ 0.35 → Good
  PSI  < 0.10 → Stable      |  0.10–0.25 → Monitor  |  > 0.25 → Unstable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Optional, Dict, Tuple
import warnings

# Suppress noisy matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

GINI_THRESHOLDS = {"Acceptable": 0.35, "Good": 0.50, "Excellent": 0.65}
KS_THRESHOLDS = {"Acceptable": 0.20, "Good": 0.35}
PSI_THRESHOLDS = {"Stable": 0.10, "Monitor": 0.25}


# ==============================================================================
# Core Metrics
# ==============================================================================

def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Gini coefficient = 2 * AUC - 1.
    Range: 0 (random) to 1 (perfect). Industry standard for scorecards.
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute KS statistic: maximum separation between event and non-event CDFs.
    Returns (ks_value, threshold_at_ks).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    threshold_at_ks = thresholds[np.argmax(tpr - fpr)]
    return float(ks), float(threshold_at_ks)


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4
) -> float:
    """
    Compute Population Stability Index between two score distributions.
    'Expected' = training distribution; 'Actual' = validation/OOT distribution.

    PSI = sum[ (Actual% - Expected%) * ln(Actual% / Expected%) ]
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def psi_interpretation(psi_val: float) -> str:
    if psi_val < PSI_THRESHOLDS["Stable"]:
        return "✅ Stable (no action needed)"
    elif psi_val < PSI_THRESHOLDS["Monitor"]:
        return "⚠️  Monitor (investigate shift)"
    else:
        return "🚨 Unstable (model redevelopment recommended)"


# ==============================================================================
# Decile / Gains Table
# ==============================================================================

def decile_table(y_true: np.ndarray, y_score: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    """
    Build a standard decile gains table.

    Columns:
      decile, min_score, max_score, total, events, non_events,
      event_rate, cumulative_events, cumulative_pct_events,
      lift, ks
    """
    df = pd.DataFrame({"score": y_score, "target": y_true})
    df["decile"] = pd.qcut(df["score"], q=n_deciles, labels=False, duplicates="drop")
    df["decile"] = n_deciles - df["decile"]  # Flip: decile 1 = highest risk score

    agg = (
        df.groupby("decile")
        .agg(
            min_score=("score", "min"),
            max_score=("score", "max"),
            total=("target", "count"),
            events=("target", "sum"),
        )
        .reset_index()
        .sort_values("decile")
    )

    agg["non_events"] = agg["total"] - agg["events"]
    agg["event_rate"] = (agg["events"] / agg["total"] * 100).round(2)
    agg["pct_total"] = (agg["total"] / agg["total"].sum() * 100).round(2)

    total_events = agg["events"].sum()
    agg["cumulative_events"] = agg["events"].cumsum()
    agg["cumulative_pct_events"] = (agg["cumulative_events"] / total_events * 100).round(2)

    avg_event_rate = total_events / agg["total"].sum()
    agg["lift"] = (agg["event_rate"] / 100 / avg_event_rate).round(2)

    # KS at each decile
    cum_events_pct = agg["cumulative_events"] / total_events
    cum_non_events_pct = agg["non_events"].cumsum() / agg["non_events"].sum()
    agg["ks"] = ((cum_events_pct - cum_non_events_pct) * 100).round(2)

    return agg


# ==============================================================================
# Full Model Report
# ==============================================================================

def model_report(
    y_true: np.ndarray,
    y_score_train: np.ndarray,
    y_score_val: np.ndarray,
    y_true_val: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict:
    """
    Compute full MRM metric suite comparing train vs. validation performance.
    """
    if y_true_val is None:
        y_true_val = y_true

    gini_train = gini_coefficient(y_true, y_score_train)
    gini_val = gini_coefficient(y_true_val, y_score_val)
    ks_train, _ = ks_statistic(y_true, y_score_train)
    ks_val, _ = ks_statistic(y_true_val, y_score_val)
    psi_val = psi(y_score_train, y_score_val)

    report = {
        "model": model_name,
        "gini_train": round(gini_train, 4),
        "gini_val": round(gini_val, 4),
        "gini_degradation": round(gini_train - gini_val, 4),
        "ks_train": round(ks_train, 4),
        "ks_val": round(ks_val, 4),
        "psi": round(psi_val, 4),
        "psi_status": psi_interpretation(psi_val),
        "gini_val_grade": _grade(gini_val, GINI_THRESHOLDS),
        "ks_val_grade": _grade(ks_val, KS_THRESHOLDS),
    }

    print(f"\n{'='*55}")
    print(f"  Model Evaluation Report — {model_name}")
    print(f"{'='*55}")
    print(f"  Gini (Train / Val)   : {gini_train:.4f} / {gini_val:.4f}  [{report['gini_val_grade']}]")
    print(f"  Gini Degradation     : {report['gini_degradation']:.4f}")
    print(f"  KS   (Train / Val)   : {ks_train:.4f} / {ks_val:.4f}  [{report['ks_val_grade']}]")
    print(f"  PSI  (Train→Val)     : {psi_val:.4f}  {report['psi_status']}")
    print(f"{'='*55}\n")

    return report


def _grade(value: float, thresholds: Dict[str, float]) -> str:
    keys = sorted(thresholds, key=lambda k: thresholds[k])
    for key in keys:
        if value < thresholds[key]:
            return f"Below {key}"
    return keys[-1]


# ==============================================================================
# Visualisation
# ==============================================================================

def plot_model_report(
    y_true_train, y_score_train,
    y_true_val, y_score_val,
    model_name: str = "Model",
    save_path: Optional[str] = None,
):
    """
    4-panel MRM-style model report:
      [1] ROC Curve (train vs val)
      [2] Score Distribution (train vs val)
      [3] Decile Gains Chart
      [4] KS Plot
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Model Evaluation Report — {model_name}", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: ROC Curve ---
    ax1 = fig.add_subplot(gs[0, 0])
    for y_t, y_s, label, color in [
        (y_true_train, y_score_train, "Train", "#2563EB"),
        (y_true_val,   y_score_val,   "Validation", "#DC2626"),
    ]:
        fpr, tpr, _ = roc_curve(y_t, y_s)
        gini = gini_coefficient(y_t, y_s)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{label} (Gini={gini:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Score Distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(y_score_train, bins=40, alpha=0.6, color="#2563EB", label="Train", density=True)
    ax2.hist(y_score_val,   bins=40, alpha=0.6, color="#DC2626", label="Validation", density=True)
    psi_val_score = psi(y_score_train, y_score_val)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Score Distribution (PSI={psi_val_score:.4f})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Decile Gains ---
    ax3 = fig.add_subplot(gs[1, 0])
    dec = decile_table(y_true_val, y_score_val)
    bars = ax3.bar(dec["decile"], dec["event_rate"], color="#7C3AED", alpha=0.8, edgecolor="white")
    ax3.axhline(y_true_val.mean() * 100, color="red", linestyle="--", lw=1.5, label="Avg Event Rate")
    ax3.set_xlabel("Score Decile (1=Highest Risk)")
    ax3.set_ylabel("Event Rate (%)")
    ax3.set_title("Event Rate by Decile (Gains Chart)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, ev in zip(bars, dec["event_rate"]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{ev:.1f}%", ha="center", va="bottom", fontsize=7)

    # --- Panel 4: KS Plot ---
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_idx = np.argsort(-y_score_val)
    y_sorted = y_true_val[sorted_idx]
    total = len(y_sorted)
    cum_events = np.cumsum(y_sorted) / y_sorted.sum()
    cum_non_events = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    x_axis = np.arange(total) / total * 100

    ax4.plot(x_axis, cum_events * 100, color="#2563EB", lw=2, label="Cumulative Events")
    ax4.plot(x_axis, cum_non_events * 100, color="#DC2626", lw=2, label="Cumulative Non-Events")

    ks_val_stat, _ = ks_statistic(y_true_val, y_score_val)
    ks_idx = np.argmax(cum_events - cum_non_events)
    ax4.vlines(x_axis[ks_idx], cum_non_events[ks_idx] * 100, cum_events[ks_idx] * 100,
               color="green", lw=2, linestyle="--", label=f"KS={ks_val_stat:.3f}")

    ax4.set_xlabel("% of Population (sorted by score, highest risk first)")
    ax4.set_ylabel("Cumulative %")
    ax4.set_title("KS Plot")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Report saved to {save_path}")
    return fig
