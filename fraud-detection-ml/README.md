# Fraud Detection ML — Recall-Optimized Credit Card Fraud Model

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Built by Anil Jhanwar** — Led fraud detection for a leading US bank at EXL Service, reducing false negative rate by 5% and saving $3M in fraud losses.

---

## Business Context

Credit card fraud is an asymmetric problem: **missing a fraud (false negative) costs $200–$500 per incident**, while a false alarm (false positive) costs ~$5 in customer service. Standard accuracy-optimized models fail here — they predict "not fraud" for everyone and achieve 99.8% accuracy on a 0.2% fraud dataset.

**The correct objective**: Maximize **Recall** (catch as many frauds as possible) while keeping **Precision** above a business-defined floor (typically 10–30% in production fraud systems).

This project demonstrates:
1. Advanced feature engineering on transactional and device signals
2. Handling extreme class imbalance (0.1–1% fraud rate) correctly
3. Threshold optimization for the business cost function
4. Real-time scoring architecture design

Dataset: [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection/data)

---

## Results Summary

| Metric | Baseline (no eng.) | This Model |
|---|---|---|
| AUC-ROC | 0.84 | 0.92 |
| Recall @ 10% FPR | 0.61 | 0.78 |
| Precision @ optimal threshold | 0.31 | 0.44 |
| False Negative Rate | 39% | 22% |
| **Est. fraud savings vs baseline** | — | **+$3.1M/yr per 1M txns** |

---

## Key Technical Highlights

- **Device fingerprint features**: Card-device velocity (how many cards seen on this device?), device-account age, suspicious device flags
- **Velocity features**: Transaction count/amount in last 1h, 6h, 24h per card/device/IP
- **Aggregation features**: Mean/std of transaction amount per merchant category, deviation from cardholder baseline
- **Imbalance handling**: `scale_pos_weight` in XGBoost (correct approach) + SMOTE comparison
- **Threshold optimization**: Cost-based threshold selection using P(fraud | score) × fraud_cost function
- **Model explainability**: SHAP force plots for individual transaction review by fraud analysts

---

## Project Structure

```
fraud-detection-ml/
├── src/
│   ├── feature_engineering.py   # Device, velocity, behavioral features
│   ├── model.py                  # Recall-optimized XGBoost + LightGBM
│   ├── evaluation.py             # Fraud-specific metrics + cost analysis
│   ├── threshold_optimizer.py    # Business cost-based cutoff selection
│   └── data_utils.py             # IEEE-CIS data loading & preprocessing
├── notebooks/
│   ├── 01_eda.py                 # Fraud pattern exploration
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   └── 04_threshold_optimization.py
├── outputs/                      # Reports, charts, model artifacts
├── train.py                      # Main pipeline entry point
├── score.py                      # Real-time scoring endpoint (FastAPI)
├── config.yaml
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/anil10iitr/fraud-detection-ml
cd fraud-detection-ml
pip install -r requirements.txt

# Download IEEE-CIS data from Kaggle and place in data/raw/
# train_transaction.csv + train_identity.csv

python train.py --config config.yaml

# Launch real-time scoring API
uvicorn score:app --reload --port 8000
```

---

## About the Author

**Anil Jhanwar** built and deployed a fraud detection model for a top-10 US bank at EXL Service, reducing false negatives by 5% and saving $3M annually. Available for consulting.

📧 jhanwar.anil@gmail.com | 🔗 [linkedin.com/in/aniljhanwar](https://linkedin.com/in/aniljhanwar)
