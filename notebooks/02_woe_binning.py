# %% [markdown]
# # WOE Binning & Feature Selection Walkthrough
# **Author:** Anil Jhanwar
#
# Weight of Evidence (WOE) is the standard feature engineering approach for
# regulatory scorecards. This notebook demonstrates:
# 1. Fitting WOE bins for each feature
# 2. Visualizing WOE monotonicity
# 3. IV-based feature selection
# 4. Interpreting the results for an MRM review

# %% Imports
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path.cwd()
if _ROOT.name == "notebooks":
    _ROOT = _ROOT.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

from src.data_utils import load_data, preprocess, split_data
from src.woe_encoder import WOEBinEncoder

plt.style.use("seaborn-v0_8-whitegrid")

# %% Load & prep data
df = load_data("data/raw/cs-training.csv")
df_clean = preprocess(df)
X_train, X_val, X_oot, y_train, y_val, y_oot = split_data(df_clean)

# %% Fit WOE encoder
encoder = WOEBinEncoder(n_bins=10, min_bin_pct=0.05, enforce_monotone=True)
encoder.fit(X_train, y_train)

# %% IV Summary
iv_summary = encoder.get_iv_summary(threshold=0.02)
print("=== Information Value Summary ===")
print(iv_summary.to_string(index=False))

# Visualize IV
fig, ax = plt.subplots(figsize=(10, 5))
colors = iv_summary["iv"].apply(lambda x:
    "#DC2626" if x >= 0.3 else "#2563EB" if x >= 0.1 else
    "#F59E0B" if x >= 0.02 else "#94A3B8"
)
ax.barh(iv_summary["feature"], iv_summary["iv"], color=colors)
ax.axvline(0.02, color="gray", linestyle="--", label="Threshold (0.02)")
ax.axvline(0.10, color="orange", linestyle="--", label="Medium (0.10)")
ax.axvline(0.30, color="red", linestyle="--", label="Strong (0.30)")
ax.set_xlabel("Information Value")
ax.set_title("Feature Selection by Information Value")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/woe_iv_summary.png", dpi=150)

# %% Visualize WOE bins for top 3 features
top_features = iv_summary.head(3)["feature"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, feature in enumerate(top_features):
    bt = encoder.get_bin_table(feature)
    bt_clean = bt[bt["bin"] != "MISSING"]
    axes[i].bar(range(len(bt_clean)), bt_clean["woe"], color="#7C3AED", alpha=0.8)
    axes[i].axhline(0, color="black", lw=1)
    axes[i].set_title(f"{feature}\nIV={encoder.iv_[feature]:.3f}", fontsize=9)
    axes[i].set_xlabel("Bin")
    axes[i].set_ylabel("WOE")
    axes[i].set_xticks(range(len(bt_clean)))
    axes[i].set_xticklabels(
        [f"B{j+1}" for j in range(len(bt_clean))], fontsize=7
    )

plt.suptitle("WOE Profiles — Top 3 Features (Monotone Enforced)", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/woe_bin_profiles.png", dpi=150)

# %% Print full bin table for top feature
top_feature = top_features[0]
print(f"\n=== Full Bin Table: {top_feature} ===")
bt = encoder.get_bin_table(top_feature)
print(bt[["bin", "count", "events", "event_rate", "woe", "iv_contribution"]].to_string(index=False))

# %% Transform and verify
X_train_woe = encoder.transform(X_train)
selected = encoder.selected_features(threshold=0.02)
print(f"\nSelected {len(selected)} features for model: {selected}")
print("\nWOE transformed data (first 3 rows):")
print(X_train_woe[selected].head(3))
