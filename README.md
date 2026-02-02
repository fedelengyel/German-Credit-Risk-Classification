# German-Credit-Risk-Classification
Comparativa de modelos de Machine Learning (Random Forest vs Logistic Regression) para la detección de riesgo crediticio, optimizando el umbral de decisión para reducir pérdidas financieras.

# Overview
This project implements a full machine learning pipeline to assess credit risk and support automated loan approval decisions. By comparing Logistic Regression (interpretable baseline) and Random Forest (non-linear model), the goal is to identify the best approach to balance risk mitigation with business growth.
Beyond classification, the project provides actionable insights for credit policy design through threshold optimization and feature importance analysis.

The model doesn't just classify applicants—it provides actionable insights for credit policy design through threshold optimization.

# Business Problem
Challenge: Financial institutions face a critical trade-off when evaluating loan applications.

Approve too liberally → Higher default rates, significant financial losses
Reject too conservatively → Lost revenue opportunities, competitive disadvantage

Solution: A data-driven credit scoring model that:
- Predicts default probability for each applicant
- Allows threshold tuning based on risk appetite
- Quantifies the business impact of different approval policies

Impact:
- 76% of risky clients filtered at 0.5 threshold
- Flexibility to adjust between risk aversion (0.67 threshold) and volume optimization (0.5 threshold)

# Dataset
Source: German Credit Data (via: Kaggle)
- 1,000 observations
- Financial and demographic features
- Target variable: Risk (good / bad)

# Methodology
- Data cleaning and preprocessing
- Explicit treatment of missing values as informative categories
- Feature engineering and encoding of categorical variables
- Train / test split with stratification
- Models implemented:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (non-linear model)
- Model evaluation using:
  - ROC-AUC
  - Precision, Recall, Accuracy
  - Confusion Matrix
- Analysis of different decision thresholds to simulate real credit policies

# Results
- Logistic Regression provides interpretability and stable baseline performance
- Random Forest achieves better class separation (ROC-AUC ≈ 0.77)
- Increasing the decision threshold reduces approved risky clients at the cost of lower approval volume
- Most important features include:
  - Checking account status
  - Loan duration
  - Credit amount
  - Age

# Business Insight
Model performance is evaluated not only through global metrics, but also through its economic impact.
Adjusting the decision threshold allows balancing risk control and loan approval volume,
which reflects real-world credit risk management.
* The Choice: I selected Random Forest with a 0.5 threshold as the optimal choice. It provides a balanced trade-off: high precision in approvals while effectively filtering out 76% of risky clients.
* Risk Policy: A higher threshold (0.67) could be adopted if the bank's priority shifts toward extreme risk aversion, though at the cost of lower approval volume.

# Technologies
- Python
- pandas, numpy
- scikit-learn
- Machine Learning
- Credit Risk Modeling
