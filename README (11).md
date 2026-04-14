# Credit Risk Prediction Using Machine Learning

### Loan Prediction Dataset --- Classification Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Platform](https://img.shields.io/badge/Platform-Jupyter%20Notebook-yellow?logo=jupyter)
![Type](https://img.shields.io/badge/Type-Supervised%20Learning-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

------------------------------------------------------------------------

## Project Overview

This project applies **supervised machine learning** to predict whether
a loan applicant is likely to default. The goal is to help financial
institutions reduce risk by making smarter lending decisions.

> **Business Problem:** Can we identify high-risk applicants before
> approving loans to minimize financial losses?

------------------------------------------------------------------------

## Objectives

-   Perform **Exploratory Data Analysis (EDA)**
-   Handle missing values and preprocess data
-   Build classification models
-   Evaluate model performance using metrics
-   Identify key factors affecting loan approval

------------------------------------------------------------------------

## Project Structure

    credit-risk-prediction/
    ├── credit_risk_prediction.ipynb
    ├── README.md
    └── dataset.csv

------------------------------------------------------------------------

## Dataset Details

  Property    Value
  ----------- ---------------------------------------
  Dataset     Loan Prediction Dataset
  Task Type   Classification
  Target      Loan Status (Approved / Not Approved)

### Features

  Feature           Description
  ----------------- -----------------------------------
  ApplicantIncome   Income of applicant
  LoanAmount        Loan amount requested
  Education         Applicant education
  Credit_History    Credit history (important factor)
  Employment        Employment status

------------------------------------------------------------------------

## ML Pipeline

    Load Dataset
         ↓
    Data Cleaning
         ↓
    EDA (Visualization & Analysis)
         ↓
    Feature Encoding
         ↓
    Train-Test Split
         ↓
    Model Training
         ↓
    Evaluation

------------------------------------------------------------------------

## Models Used

### Logistic Regression

-   Simple and interpretable model
-   Works well for binary classification

### Decision Tree

-   Captures non-linear relationships
-   Easy to visualize and interpret

------------------------------------------------------------------------

## Model Evaluation

-   Accuracy Score
-   Confusion Matrix

------------------------------------------------------------------------

## Results

-   Credit History is the most important feature
-   Income and Loan Amount also impact predictions
-   Model achieves good prediction performance

------------------------------------------------------------------------

## Visualizations

-   Income Distribution
-   Loan Amount Distribution
-   Education vs Loan Status
-   Confusion Matrix

------------------------------------------------------------------------

## Dependencies

``` txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install using:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

------------------------------------------------------------------------

## Skills Demonstrated

  Skill              Description
  ------------------ ---------------------------------
  Data Cleaning      Handling missing values
  EDA                Data visualization and insights
  Machine Learning   Classification models
  Evaluation         Accuracy and confusion matrix

------------------------------------------------------------------------

## Future Improvements

-   Use Random Forest / XGBoost
-   Hyperparameter tuning
-   Deploy model using Flask or Streamlit

------------------------------------------------------------------------




