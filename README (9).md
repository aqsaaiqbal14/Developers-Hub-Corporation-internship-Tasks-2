#  Customer Segmentation Using Unsupervised Learning
### Mall Customers Dataset — K-Means + PCA + t-SNE

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![Unsupervised](https://img.shields.io/badge/Type-Unsupervised%20Learning-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Project Overview

This project applies **unsupervised machine learning** to segment mall customers into distinct groups based on their **age**, **annual income**, and **spending score**. Each segment is then analyzed and assigned a **targeted marketing strategy** to help the business maximize revenue and customer satisfaction.

> **Business Problem:** Not all customers are the same. Instead of one-size-fits-all marketing, can we identify *who* our customers really are — and market to each group differently?

---

##  Objectives

-  Conduct thorough Exploratory Data Analysis (EDA)
-  Apply **K-Means Clustering** to segment customers
-  Use **Elbow Method** + **Silhouette Score** to find the optimal number of clusters
-  Use **PCA** (Principal Component Analysis) to visualize clusters in 2D
-  Use **t-SNE** for non-linear dimensionality reduction and cluster visualization
-  Propose **targeted marketing strategies** for each customer segment

---

##  Project Structure

```
 customer-segmentation/
├──  customer_segmentation.ipynb    ← Main Colab notebook (run this)
├──  README.md                      ← You are here
└──  Mall_Customers.csv             ← Dataset (upload when prompted in Colab)
```

---

##  Dataset Details

| Property  | Value |
|-----------|-------|
| Source    | [Kaggle — Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| File      | `Mall_Customers.csv` |
| Rows      | 200 customers |
| Columns   | 5 |
| Task Type | Unsupervised Clustering |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `CustomerID` | ID | Unique customer identifier |
| `Gender` | Categorical | Male / Female |
| `Age` | Numerical | Customer age (18–70) |
| `Annual_Income` | Numerical | Annual income in thousands (k$) |
| `Spending_Score` | Numerical | Mall-assigned score (1–100) based on spending behavior |

---

##  ML Pipeline — Step by Step

```
 Load & Rename Columns
        ↓
 EDA
   → Distributions, Gender comparisons, Boxplots
   → Scatter plots (Income vs Spending)
   → Correlation Heatmap + Pairplot
        ↓
  Preprocessing
   → Label Encode Gender
   → StandardScaler on [Age, Annual_Income, Spending_Score]
        ↓
 Find Optimal K
   → Elbow Method (Inertia for K=2 to 10)
   → Silhouette Score (K=2 to 10)
        ↓
 K-Means Clustering (K=5)
   → k-means++ initialization, 20 restarts
        ↓
 Cluster Profiling
   → Stats table, Bar charts, 2D scatter, Bubble chart
        ↓
 PCA (2 Components)
   → Scree plot, Cluster view, Feature loadings
        ↓
 t-SNE (2 Components)
   → Cluster view, Gender view, PCA vs t-SNE comparison
        ↓
 Marketing Strategies per Segment
   → Strategy bubble map
        ↓
 Final Summary Table
```

---

##  Algorithms Used

### K-Means Clustering
- Partitions customers into **K non-overlapping groups**
- Uses **k-means++ initialization** for better convergence
- Parameters: `n_clusters=5`, `n_init=20`, `max_iter=500`, `random_state=42`

### PCA — Principal Component Analysis
- **Linear** dimensionality reduction: 3 features → 2 principal components
- Visualizes cluster separation in 2D
- Also produces a **feature loading plot** showing each feature's contribution per component

### t-SNE — t-Distributed Stochastic Neighbor Embedding
- **Non-linear** dimensionality reduction, preserves local neighborhood structure
- Often produces cleaner cluster separation than PCA
- Parameters: `perplexity=30`, `n_iter=1000`, `init='pca'`, `random_state=42`

---

##  Finding the Optimal K

| Method | How It Works | Result |
|---|---|---|
| **Elbow Method** | Plot inertia vs K — look for the "elbow" bend | Elbow at K = 5 |
| **Silhouette Score** | Measures how well-separated clusters are (−1 to 1) | Best score at K = 5 |

Both methods agreed on **K = 5** as the optimal number of clusters.

---

##  The 5 Customer Segments

| Cluster | Segment Name | Income | Spending | Key Trait |
|---|---|---|---|---|
| 0 |  Cautious High Earners | High | Low | Earn well, spend conservatively |
| 1 |  VIP Spenders | High | High | Most valuable customers |
| 2 |  Average Joes | Medium | Medium | Largest, most common group |
| 3 |  Impulsive Shoppers | Low | High | Love shopping despite low income |
| 4 |  Budget Conscious | Low | Low | Price-sensitive, rarely purchase |

---

##  Marketing Strategies per Segment

###  Cluster 0 — Cautious High Earners
> *High income, low spending — huge untapped potential*

- Offer exclusive **premium loyalty programs** to incentivize spending
- Promote **luxury or investment products** (wealth management, premium cards)
- Send personalized **"value for money"** deals to trigger first purchases
- Focus messaging on **quality and exclusivity**, not discounts

###  Cluster 1 — VIP Spenders
> *High income + high spending — the most valuable segment*

- Offer **VIP membership**, early product access, and exclusive rewards
- **Upsell** premium and high-margin products aggressively
- Provide **dedicated personal shoppers** or concierge service
- Retain with **surprise gifts**, loyalty points, and milestone rewards

###  Cluster 2 — Average Joes
> *Middle income + middle spending — the core customer base*

- Promote **bundles and combo deals** for better perceived value
- Run **seasonal sales**, flash deals, and limited-time offers
- Use **referral programs** (refer-a-friend) to grow reach organically
- Email campaigns with **personalized product recommendations**

###  Cluster 3 — Impulsive Shoppers
> *Low income + high spending — they love to shop*

- Offer easy **EMI / Buy-Now-Pay-Later (BNPL)** installment options
- Push frequent **discounts, flash sales**, and coupon codes
- **Gamify shopping** with spin-to-win rewards and daily check-ins
- Promote **affordable trending products** and social-media-viral items

###  Cluster 4 — Budget Conscious
> *Low income + low spending — hardest to convert*

- Target with **deep discounts**, clearance sales, and value packs
- Introduce a **free-tier or trial membership** to build engagement
- Send **re-engagement campaigns** ("We miss you!") with exclusive coupons
- Focus exclusively on **essential, everyday low-cost products**

---

##  Dimensionality Reduction Summary

| Technique | Type | Preserves | Best For |
|---|---|---|---|
| **PCA** | Linear | Global variance | Quick overview, interpretable loadings |
| **t-SNE** | Non-linear | Local structure | Revealing tight cluster boundaries |

> Both are used **only for visualization** — K-Means runs on the original scaled features.

---

##  Visualizations Produced (20+ Charts)

| # | Chart | Purpose |
|---|---|---|
| 1 | Histograms (3 features) | Feature distributions |
| 2 | Gender-wise histograms | Male vs Female patterns |
| 3 | Boxplots by Gender | Spread and outliers |
| 4 | Scatter: Income vs Spending (by Gender) | Raw cluster structure |
| 5 | Scatter: Income vs Spending (by Age Group) | Age-based patterns |
| 6 | Correlation Heatmap | Feature relationships |
| 7 | Pairplot | All feature combinations by gender |
| 8 | Elbow Curve | Find optimal K |
| 9 | Silhouette Bar Chart | Confirm optimal K |
| 10 | Cluster Scatter (Income vs Spending) | Final clusters with centroids |
| 11 | Bubble Chart (Age vs Income) | 3D-style cluster view |
| 12 | Cluster Profile Bar Charts | Compare clusters on all metrics |
| 13 | PCA Scree + Cumulative Variance | Variance explained per component |
| 14 | PCA — Cluster View | 2D cluster separation |
| 15 | PCA — Gender View | Gender distribution in PCA space |
| 16 | PCA Feature Loadings | Feature contributions to PC1 & PC2 |
| 17 | t-SNE — Cluster View | Non-linear cluster separation |
| 18 | t-SNE — Gender View | Gender in t-SNE space |
| 19 | PCA vs t-SNE Comparison | Side-by-side method comparison |
| 20 | Marketing Strategy Bubble Map | Business strategy visualization |

---

> **Note for local run:** Replace the `files.upload()` cell with:
> ```python
> df = pd.read_csv('Mall_Customers.csv')
> ```

---

##  Dependencies

All libraries are **pre-installed in Google Colab** — no extra installations needed.

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install locally all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

##  Skills Demonstrated

| Skill | Details |
|---|---|
| **Unsupervised Learning** | K-Means clustering with k-means++ |
| **Optimal Cluster Selection** | Elbow Method + Silhouette Score |
| **Dimensionality Reduction** | PCA (linear) + t-SNE (non-linear) |
| **Feature Engineering** | StandardScaler, Label Encoding, Age Group binning |
| **Customer Segmentation** | 5 distinct behavioral segments identified |
| **Strategy Development** | Data-driven marketing recommendations per segment |
| **Data Visualization** | 20+ charts — Matplotlib + Seaborn |
| **EDA** | Missing values, distributions, correlations, pairplots |

---


##  Acknowledgements

- Dataset: [Kaggle — Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Scikit-learn Documentation](https://scikit-learn.org) — KMeans, PCA, TSNE
- t-SNE Paper: van der Maaten & Hinton (2008) — [JMLR](https://jmlr.org/papers/v9/vandermaaten08a.html)
