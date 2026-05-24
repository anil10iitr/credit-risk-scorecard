# Model Card — Credit Application Scorecard

**Version**: 1.0  
**Last Updated**: 2025  
**Author**: Anil Jhanwar  
**Model Type**: Credit Risk Scorecard (Champion: Logistic Regression | Challenger: XGBoost)

---

## 1. Model Intent & Use

**Intended Use**: Score consumer loan/credit card applicants at point-of-application to predict probability of serious delinquency (90+ days past due) within 24 months.

**Out-of-Scope Uses**:
- Not designed for small business or commercial lending
- Not for post-acquisition account management (use collections model instead)
- Not validated for populations outside the training region/period

---

## 2. Training Data

| Attribute | Detail |
|---|---|
| Source | Give Me Some Credit (Kaggle / representative of US consumer bureau data) |
| Rows | ~150,000 applicants |
| Target | SeriousDlqin2yrs (1 = defaulted, 0 = performed) |
| Target Rate | ~6.7% |
| Train Period | Rows 1–70% (stratified) |
| Validation | Rows 71–85% (stratified) |
| OOT | Rows 86–100% (out-of-time holdout) |

---

## 3. Features & Information Value

| Feature | IV | Interpretation |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 0.42 | Strong |
| NumberOfTimes90DaysLate | 0.38 | Strong |
| NumberOfTime30-59DaysPastDueNotWorse | 0.28 | Medium |
| NumberOfTime60-89DaysPastDueNotWorse | 0.21 | Medium |
| age | 0.18 | Medium |
| DebtRatio | 0.09 | Weak |
| MonthlyIncome | 0.07 | Weak |

---

## 4. Performance Metrics

| Metric | Train | Validation | OOT |
|---|---|---|---|
| Gini | 0.67 | 0.64 | 0.62 |
| KS | 0.44 | 0.41 | 0.40 |
| AUC | 0.84 | 0.82 | 0.81 |
| PSI (vs Train) | — | 0.04 | 0.06 |

All PSI values < 0.10 → **Stable model, no population shift.**

---

## 5. Scorecard Scaling Parameters

- PDO (Points to Double Odds): **20**
- Base Score: **600** (industry standard)
- Base Odds: **30:1** (good:bad at score 600)
- Score Range: **300–850** (capped to FICO range)
- **Recommended Cutoff**: Score < 580 → Decline | Score 580–620 → Review | Score > 620 → Approve

---

## 6. Limitations & Risks

- **Temporal Drift**: Model trained on pre-2020 data; requires PSI monitoring if deployed post-2023
- **Class Imbalance**: 6.7% bad rate; threshold tuning required for each deployment context
- **Missing Income**: Monthly income has ~20% missing rate; median imputation may underperform for high-income applicants
- **Geographic Scope**: Trained on US consumer data; not validated for SEA or India without recalibration

---

## 7. Monitoring Recommendations

| Metric | Frequency | Alert Threshold | Action |
|---|---|---|---|
| PSI | Monthly | > 0.10 | Investigate; redevelop if > 0.25 |
| Gini (OOT) | Quarterly | Drop > 0.05 vs baseline | Redevelop |
| Bad Rate by Decile | Monthly | ±20% from baseline | Recalibrate cutoffs |

---

## 8. Ethical Considerations

- Age is used as a predictor — ensure compliance with ECOA/Fair Lending regulations in US deployments
- Model should be tested for disparate impact across protected classes before production deployment
- Recommend quarterly fairness audit: approval rates by demographic segment
