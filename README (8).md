#  Term Deposit Subscription Prediction
### Bank Marketing Dataset — Full Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-red)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Project Overview

This project builds a **binary classification system** to predict whether a bank customer will subscribe to a **term deposit** as a result of a telemarketing campaign.

The dataset is the well-known **UCI Bank Marketing Dataset** (`bank-full.csv`) containing **45,211 records** and **17 features** from a Portuguese bank's phone-based marketing campaigns.

> **Business Problem:** Instead of calling every customer, can we predict *who* is likely to say YES — saving time and resources for the bank?

---

##  Objectives

-  Load and explore the dataset (EDA)
-  Encode all categorical features properly (Label + One-Hot Encoding)
-  Handle class imbalance using **SMOTE**
-  Train classification models: **Logistic Regression** and **Random Forest**
-  Evaluate models using **Confusion Matrix**, **F1-Score**, and **ROC Curve**
-  Explain predictions using **SHAP** (Explainable AI) for 5+ individual customers

---

##  Project Structure

```
bank-marketing-prediction/
├──  bank_marketing_prediction.ipynb   ← Main Colab notebook (run this)
├──  README.md                         ← You are here
└──  bank-full.csv                     ← Dataset (upload when prompted in Colab)
```

---

##  Dataset Details

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| File | `bank-full.csv` |
| Separator | Semicolon (`;`) |
| Rows | 45,211 |
| Features | 16 input + 1 target (`y`) |
| Target | `yes` = subscribed, `no` = not subscribed |
| Class Balance | ~88% No / ~12% Yes (imbalanced) |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the customer |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Binary | Has credit in default? |
| `balance` | Numerical | Average yearly balance (euros) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `day` | Numerical | Last contact day of month |
| `month` | Categorical | Last contact month |
| `duration` | Numerical | Last contact duration (seconds) |
| `campaign` | Numerical | Number of contacts in this campaign |
| `pdays` | Numerical | Days since last contact from previous campaign |
| `previous` | Numerical | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign |
| `y` | **Target** | Has client subscribed? (yes/no) |

---

##  ML Pipeline — Step by Step

```
 Load Data
    ↓
 EDA (shape, types, missing values, distributions, correlations)
    ↓
  Encode Categorical Features
    → Label Encoding  : default, housing, loan, y
    → One-Hot Encoding: job, marital, education, contact, month, poutcome
    ↓
  Train-Test Split (80% / 20%, stratified)
    ↓
  SMOTE — Oversample minority class (yes) in training set only
    ↓
 StandardScaler — Scale features for Logistic Regression
    ↓
 Train Models
    → Model 1: Logistic Regression
    → Model 2: Random Forest (200 trees)
    ↓
 Evaluate
    → Confusion Matrix
    → Classification Report (Precision, Recall, F1)
    → ROC Curve + AUC Score
    → Feature Importance (Random Forest)
    ↓
 SHAP Explainability
    → Global Summary Plot (bar)
    → Beeswarm Plot
    → 5 Individual Waterfall Charts
    → Force Plot
    ↓
 Final Model Comparison Summary + Bar Chart
```

---

##  Models Used

### 1. Logistic Regression
- Simple, interpretable baseline model
- Trained on **SMOTE-balanced** + **StandardScaler-normalized** data
- `max_iter=1000`, `random_state=42`

### 2. Random Forest Classifier
- Powerful ensemble model (200 decision trees)
- Does not require feature scaling
- `n_estimators=200`, `max_depth=15`, `random_state=42`, `n_jobs=-1`

---

##  Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| **Confusion Matrix** | Shows True/False Positives and Negatives visually |
| **F1-Score** | Balances Precision & Recall — ideal for imbalanced datasets |
| **ROC-AUC** | Measures how well the model separates YES vs NO customers |
| **Precision** | Of all predicted YES — how many actually said YES? |
| **Recall** | Of all actual YES — how many did we correctly catch? |

>  Accuracy alone is misleading here because 88% of customers say NO — a model predicting NO every time would get 88% accuracy but be useless!

---

##  Explainability with SHAP

**SHAP (SHapley Additive exPlanations)** answers: *"Why did the model predict YES or NO for this specific customer?"*

| Plot Type | What It Shows |
|---|---|
| **Bar Summary** | Most important features globally across all predictions |
| **Beeswarm** | Direction + magnitude of each feature's impact |
| **Waterfall** | Step-by-step explanation for 1 individual customer |
| **Force Plot** | Interactive visual of feature contributions |

### Key Findings from SHAP:
-  **`duration`** (call length) is the strongest predictor — longer calls → higher chance of YES
-  **`poutcome`** (previous campaign result) is highly influential
-  **`balance`** and **`age`** also play important roles
-  **`month`** of contact affects subscription rates significantly

---

##  Dependencies

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
shap >= 0.40
imbalanced-learn
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap imbalanced-learn
```

---

##  Known Issues & Fixes Applied

### 1. SHAP AssertionError — Shape Mismatch
**Problem:** Old SHAP API `explainer.shap_values(X)[1]` returns wrong shape in SHAP ≥ 0.40  
**Fix:** Use modern API: `explainer(X)[:, :, 1]` which returns a proper `Explanation` object

### 2. Waterfall Charts Overlapping
**Problem:** Using `plt.subplots(5,1)` with `plt.sca()` causes SHAP to draw all plots on top of each other  
**Fix:** Each waterfall gets its own `fig, ax = plt.subplots()` with `plt.show()` called immediately after

---

##  Skills Demonstrated

| Skill | Details |
|---|---|
| **Classification Modeling** | Logistic Regression, Random Forest |
| **Feature Encoding** | Label Encoding, One-Hot Encoding |
| **Imbalanced Data Handling** | SMOTE oversampling |
| **Model Evaluation** | Confusion Matrix, F1, ROC-AUC, Precision, Recall |
| **Explainable AI (XAI)** | SHAP — global + individual prediction explanations |
| **Customer Behavior Analysis** | Feature importance, business insights from model |
| **Data Visualization** | Seaborn, Matplotlib — 10+ charts |

---


##  Acknowledgements

- Dataset: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- S. Moro, P. Cortez and P. Rita — *"A Data-Driven Approach to Predict the Success of Bank Telemarketing"* (2014)
- [SHAP Library](https://github.com/slundberg/shap) by Scott Lundberg
