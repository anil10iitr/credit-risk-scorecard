# Credit Risk Scorecard — End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Built by Anil Jhanwar** — AI/ML Consultant | 15+ years in Credit Risk & Fintech  
> Modeled after production systems deployed at scale in major Indian banks and Southeast Asian fintechs.

---

## Business Context

Credit scorecards are the backbone of consumer lending — they determine who gets a loan, at what interest rate, and with what credit limit. A 1% improvement in model Gini can translate directly to **₹50–500M in reduced NPAs** for a mid-size bank.

This project replicates a **production-grade credit application scoring pipeline** using the publicly available [Give Me Some Credit dataset (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit/data), demonstrating the full model development lifecycle from raw data to a deployable scorecard with regulator-ready documentation.

---

## Results Summary

| Metric | Logistic Scorecard | XGBoost |
|---|---|---|
| Gini Coefficient | 0.64 | 0.72 |
| KS Statistic | 0.41 | 0.49 |
| AUC-ROC | 0.82 | 0.86 |
| PSI (train vs val) | 0.04 | 0.06 |

> PSI < 0.10 = stable model. Both models pass MRM stability checks.

---

## Pipeline Architecture

```
Raw Data
   ↓
01_eda.py            → Missingness, distributions, target rate analysis
   ↓
02_woe_binning.py    → Weight of Evidence encoding, IV filtering
   ↓
03_model_training.py → Logistic Regression scorecard + XGBoost challenger
   ↓
04_evaluation.py     → Gini, KS, PSI, decile table, SHAP explanations
   ↓
model_card.md        → Regulator-ready model documentation
```

---

## Project Structure

```
credit-risk-scorecard/
├── src/
│   ├── woe_encoder.py        # Custom WOE binning with monotonicity enforcement
│   ├── scorecard_model.py    # Full scorecard pipeline (champion + challenger)
│   ├── evaluation.py         # Gini, KS, PSI, decile analysis
│   └── data_utils.py         # Data loading, validation, train/val/OOT split
├── notebooks/
│   ├── 01_eda.py             # Exploratory analysis (run as Jupyter via Jupytext)
│   ├── 02_woe_binning.py     # WOE/IV feature selection walkthrough
│   ├── 03_model_training.py  # Champion/challenger model training
│   └── 04_evaluation.py      # Full model evaluation suite
├── outputs/
│   ├── scorecard_points.csv  # Final scorecard with points per variable
│   ├── decile_table.csv       # Gains/lift table
│   └── model_report.html      # Auto-generated HTML report
├── train.py                  # Main entry point — run full pipeline
├── predict.py                # Inference on new applicants
├── model_card.md             # Model documentation for MRM/regulators
├── config.yaml               # All hyperparameters and thresholds
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/anil10iitr/credit-risk-scorecard
cd credit-risk-scorecard
pip install -r requirements.txt

# 2. Download data
# Place cs-training.csv from Kaggle into data/raw/

# 3. Run full pipeline
python train.py --config config.yaml

# 4. Score new applicants
python predict.py --input data/new_applicants.csv --output outputs/scores.csv
```

---

## Key Technical Highlights

- **WOE Binning**: Custom implementation with monotonicity enforcement and fine/coarse classing — not just a library wrapper
- **IV Filtering**: Automatic feature selection based on Information Value (IV > 0.02)
- **Scorecard Scaling**: Points-to-double-odds scaling (PDO=20, base score=600, base odds=30:1) — standard for FICO-style cards
- **Champion/Challenger**: Both logistic regression (interpretable, regulator-preferred) and XGBoost (performance) trained and evaluated
- **PSI Monitoring**: Population Stability Index computed between train, validation, and out-of-time (OOT) sets
- **SHAP Explanations**: Force plots and summary plots for model explainability

---

## Model Card

See [`model_card.md`](model_card.md) for full MRM-ready documentation including intended use, limitations, bias analysis, and monitoring thresholds.

---

## About the Author

**Anil Jhanwar** has deployed production credit scoring systems at Large Banks (₹800M acquisition impact) and Fintechs in the US, India and South East Asia. Available for consulting engagements in credit risk, fraud, and collections AI.

📧 jhanwar.anil@gmail.com | 🔗 [linkedin.com/in/aniljhanwar](https://linkedin.com/in/aniljhanwar)
