# DA5401-A6-Imputation-via-Regression-for-Missing-Data
## Author: Dikshank Sihag (DA25M009)

# 🧭 Credit Risk Modeling under Missing Data: Imputation Strategies & Performance Evaluation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📘 Executive Summary

This project explores different imputation strategies for handling Missing at Random (MAR) data in the UCI Credit Card Default Clients dataset and evaluates their impact on Logistic Regression classification performance.

I compare:

- Model A: Median Imputation  
- Model B: Linear Regression Imputation  
- Model C: Non-Linear KNN Imputation  
- Model D: Listwise Deletion (baseline without imputation)

Each dataset was standardized and evaluated using Weighted F1, ROC-AUC, and PR-AUC — metrics suited for imbalanced binary classification.

The results show that all three imputation methods perform comparably and outperform the listwise deletion baseline in most metrics, with KNN imputation providing slightly better overall balance. Non-linear imputation better preserves feature dependencies and generalizes effectively under MAR conditions.

---

## 🧠 Theoretical Background

### Missing at Random (MAR)
Under MAR, missingness depends on other observed variables, not on the missing value itself.  
For example, older clients or those with lower `LIMIT_BAL` may have missing `BILL_AMT1`; this pattern depends on known features, not the hidden value.  

This allows imputers such as regression or KNN to estimate missing values from relationships among available features, making MAR a realistic assumption for financial data.


---

## ⚙️ Implementation Overview

### Part A – Data Preprocessing & Imputation

| Model | Technique | Concept | Pros | Cons |
|:------|:-----------|:--------|:-----|:----|
| A | Median Imputation | Replaces with robust central tendency | Fast, robust | Ignores correlations |
| B | Linear Regression | Predicts from linear predictors | Preserves correlations | Misses non-linearities |
| C | Non-Linear KNN | Imputes by nearest neighbors | Captures complex patterns | Slower |
| D | Listwise Deletion | Drops rows with NaN | Simple | Data loss, bias |

---

### Part B – Model Training & Evaluation

- Classifier: Logistic Regression  
- Split: 80 % train / 20 % test  
- Scaler: StandardScaler  
- Metrics: Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | PR-AUC

---

### Part C – Results Summary

| Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | PR-AUC |
|:------|:---------:|:----------:|:-------:|:------------:|:--------:|:-------:|
| Model A (Median) | 0.8078 | 0.7894 | 0.8078 | 0.7691 | 0.7071 | 0.4935 |
| Model B (Linear Reg) | 0.8077 | 0.7887 | 0.8077 | 0.7695 | 0.7073 | 0.4936 |
| Model C (Non-Linear KNN) | 0.8077 | 0.7888 | 0.8077 | 0.7694 | 0.7074 | 0.4935 |
| Model D (Listwise Deletion) | 0.8057 | 0.7845 | 0.8057 | 0.7672 | 0.7201 | 0.4783 |

---
## 💡 Insights & Discussion

### Listwise Deletion vs Imputation
Listwise Deletion (Model D) removes data, reducing representativeness and statistical power.  
Imputation-based models (A–C) retain all samples, allowing stronger generalization and less bias.  
Model D performs slightly worse overall because it discards informative patterns during training.

### Linear vs Non-Linear Imputation
Linear Regression Imputation (Model B) preserves basic correlations but assumes linear relationships.  
Non-Linear KNN Imputation (Model C) captures local non-linear patterns, modeling realistic credit behavior.  
Their near-identical performance suggests that under mild MAR conditions, both linear and non-linear imputers perform consistently, though KNN offers more flexibility in complex datasets.

### Recommended Strategy

| Case | Recommended Approach | Rationale |
|------|----------------------|------------|
| MCAR | Simple Median/Mean | No systematic bias |
| MAR | KNN / MissForest / Iterative RF | Captures observed dependencies |
| MNAR | Specialized model-based | Depends on unobserved causes |

---

## 🧰 Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/<your-username>/UCI-Credit-Imputation.git
cd UCI-Credit-Imputation
