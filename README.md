# DA5401-A6-Imputation-via-Regression-for-Missing-Data

# 🧭 Credit Risk Modeling under Missing Data: Imputation Strategies & Performance Evaluation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📘 Executive Summary

This project explores different **imputation strategies** for handling **Missing at Random (MAR)** data in the **UCI Credit Card Default Clients** dataset and evaluates their impact on **Logistic Regression** classification performance.

We compare:

- **Model A:** Median Imputation  
- **Model B:** Linear Regression Imputation  
- **Model C:** Non-Linear KNN Imputation  
- **Model D:** Listwise Deletion (baseline without imputation)

Each dataset was standardized and evaluated using **Weighted F1**, **ROC-AUC**, and **PR-AUC** — metrics suited for imbalanced binary classification.

> 🧩 **Key Finding:**  
> Non-linear imputation (Model C, KNN) produces the most accurate and generalizable results — confirming that flexible, data-driven imputers outperform rigid linear methods under MAR conditions.

---

## 🧠 Theoretical Background

### 🔹 Why “Missing at Random” (MAR)?
Under MAR, missingness depends on *other observed variables*, not on the missing value itself.  
Example: older clients or those with lower `LIMIT_BAL` may have missing `BILL_AMT1`; this pattern depends on known features, not the hidden value.  

This allows imputers like regression or KNN to estimate missing values from the relationships among available features, making MAR a realistic assumption for financial data.


---

## ⚙️ Implementation Overview

### **Part A – Data Preprocessing & Imputation**

| Model | Technique | Concept | Pros | Cons |
|:------|:-----------|:--------|:-----|:----|
| **A** | Median Imputation | Replaces with robust central tendency | Fast, robust | Ignores correlations |
| **B** | Linear Regression | Predicts from linear predictors | Preserves correlations | Misses non-linearities |
| **C** | Non-Linear KNN | Imputes by nearest neighbors | Captures complex patterns | Slower |
| **D** | Listwise Deletion | Drops rows with NaN | Simple | Data loss, bias |

---

### **Part B – Model Training & Evaluation**

- Classifier: **Logistic Regression**  
- Split: 80 % train / 20 % test  
- Scaler: **StandardScaler**  
- Metrics: Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | PR-AUC

---

### **Part C – Results Summary**

| Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | PR-AUC |
|:------|:---------:|:----------:|:-------:|:------------:|:--------:|:-------:|
| **A – Median** | 0.80 | 0.76 | 0.77 | 0.76 | 0.68 | 0.45 |
| **B – Linear Reg.** | 0.80 | 0.77 | 0.78 | 0.77 | 0.69 | 0.47 |
| **C – Non-Linear KNN** | **0.81** | **0.78** | **0.79** | **0.77** | **0.71** | **0.49** |
| **D – Listwise** | 0.79 | 0.75 | 0.74 | 0.74 | 0.66 | 0.42 |

> ✅ **Model C (KNN)** performs best overall — confirming that non-linear relationships capture MAR data structure more effectively.

---

## 📊 Visualizations

### 🔹 Performance Comparison
![Performance Comparison](results/metrics_comparison.png)

### 🔹 ROC & PR Curves
| ROC Curve | PR Curve |
|------------|-----------|
| ![ROC Curve](results/roc_curves.png) | ![PR Curve](results/pr_curves.png) |

---

## 💡 Insights & Discussion

### 1️⃣ Listwise Deletion vs Imputation
- Listwise Deletion (Model D) removes data → lower power & bias.  
- Imputation (A–C) retains samples → better generalization.  
- Model D underperforms due to **information loss**.

### 2️⃣ Linear vs Non-Linear Imputation
- Linear Regression Imputation (Model B) assumes straight-line relations.  
- KNN Imputation (Model C) captures **non-linear & local** patterns — essential in financial datasets.

### 3️⃣ Recommended Strategy

| Case | Recommended Approach | Rationale |
|------|----------------------|------------|
| **MCAR** | Simple Median/Mean | No systematic bias |
| **MAR** | **KNN / MissForest / Iterative RF** | Captures observed dependencies |
| **MNAR** | Specialized model-based | Depends on unobserved causes |

---

## 🧰 Installation & Setup

### 🔸 Clone the Repository
```bash
git clone https://github.com/<your-username>/UCI-Credit-Imputation.git
cd UCI-Credit-Imputation

