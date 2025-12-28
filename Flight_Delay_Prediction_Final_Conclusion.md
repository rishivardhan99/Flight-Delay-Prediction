# Flight Delay Prediction — Final Project Conclusion

## Project Overview
This project aimed to build a reliable machine learning system to predict whether a flight would be delayed (15 minutes or more) using historical flight data enriched with operational, temporal, and weather-related features. The emphasis was not only on predictive performance, but also on **methodological correctness, interpretability, and operational decision-making**.

---

## Methodology Summary

### 1. Data Preparation
- Removed post-event and leakage-prone variables (e.g., actual delays, delay causes).
- Engineered meaningful features capturing:
  - **Operational constraints** (turnaround-related indicators).
  - **Temporal effects** (departure hour, day of week).
  - **Weather conditions** (precipitation, wind, snow).
- Handled missing values conservatively to preserve data integrity.
- Applied time-based train/test splitting to respect real-world deployment constraints.

### 2. Baseline Modeling: Logistic Regression
- Built a baseline Logistic Regression model for transparency and interpretability.
- Addressed class imbalance using **class-weighted training**.
- Tuned the decision threshold to align predictions with operational priorities.
- Interpreted coefficients to validate that learned relationships aligned with domain intuition.

This baseline established a strong, explainable reference point.

### 3. Probability Calibration (Analysis)
- Explored probability calibration to assess reliability of predicted probabilities.
- Observed that calibration improved probability honesty but reduced recall at the chosen operating threshold.
- Based on project goals (minimizing missed delays), calibration was documented but not adopted for the linear model in isolation; calibration remains recommended as part of the monitoring and post-hoc probability adjustment toolkit.

### 4. Non-Linear Modeling: Random Forest
- Introduced a Random Forest model to capture non-linear interactions.
- Used conservative hyperparameters to avoid overfitting on a large dataset.
- Achieved a substantially higher ROC–AUC, indicating better class separation.
- Performed threshold tuning to adjust the model’s conservative default behavior.

At a tuned threshold of **0.30**, the Random Forest produced desirable operational trade-offs (improved recall while keeping precision at acceptable levels).

---

## Final Model Selection (Combined approach)

**Selected approach:** **Both models retained and deployed** in a governed, decision-aware configuration — Random Forest as the primary predictive engine and Logistic Regression as an interpretable fallback / adjudicator.

### What is saved and deployed
- **Random Forest** (primary predictive model)  
  - **Decision threshold:** **0.30**  
  - **Role:** main production predictor for automated decisions and ranking.
- **Logistic Regression** (interpretable reference model)  
  - **Decision threshold:** tuned to operational preference  
  - **Role:** conservative fallback, human-facing explanations, auditing, and monitoring.

### Operational decision logic
1. Use Random Forest for primary prediction.
2. If both models agree, accept the prediction.
3. If models disagree and Logistic Regression predicts delay with high confidence, escalate conservatively.
4. Optionally compute a weighted ensemble for ranking tasks.

### Governance and monitoring
- Log disagreement cases.
- Monitor calibration, drift, and threshold performance.
- Retain model artifacts and thresholds for reproducibility.

---

## Key Insights
- **Operational features dominate** delay prediction.
- **Weather effects are secondary** to scheduling and turnaround constraints.
- **Threshold tuning is critical** for real-world performance.
- **Hybrid modeling improves safety and trust**.

---

## Limitations
- Depends on historical patterns.
- Does not explicitly model rare disruptions.
- Reduced interpretability for non-linear models.

---

## Future Work
- Evaluate gradient boosting models.
- Integrate real-time data feeds.
- Expand explainability tooling.
- Production-grade deployment with monitoring.

---

## Closing Note
This project demonstrates a principled, end-to-end applied machine learning workflow focused on decision quality and operational realism.
