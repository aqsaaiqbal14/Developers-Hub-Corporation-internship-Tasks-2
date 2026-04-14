#  Energy Consumption Time Series Forecasting

**Household Power Consumption — ARIMA + Prophet + XGBoost**

---

##  Overview

This project forecasts short-term **household energy consumption** using historical time-series data. Three machine learning and statistical models are trained and compared: **ARIMA**, **Facebook Prophet**, and **XGBoost**.

The dataset used is the [UCI Household Power Consumption dataset](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption), which contains ~2 million rows of 1-minute interval readings from December 2006 to November 2010.

---

##  Project Structure

```
energy_forecasting.ipynb       # Main notebook (all steps included)
household_power_consumption.csv  # Dataset (download separately — see below)
```

---

##  Pipeline

| Step | Description |
|------|-------------|
| 1 | Install required libraries |
| 2 | Import libraries and configure styles |
| 3 | Upload & load the dataset |
| 4 | Parse datetime, resample to hourly/daily, handle missing values |
| 5 | Exploratory Data Analysis (EDA) |
| 6 | Feature engineering (time-based + cyclical + lag + rolling features) |
| 7 | Train/Test split (last 30 days for ARIMA/Prophet, last 7 days for XGBoost) |
| 8 | Model 1 — ARIMA (Auto order selection via `pmdarima`) |
| 9 | Model 2 — Facebook Prophet (yearly + weekly seasonality) |
| 10 | Model 3 — XGBoost (supervised regression on engineered features) |
| 11 | Model comparison with metrics table and bar charts |
| 12 | Final summary with key insights and recommendations |

---

##  Dependencies

Install all required libraries before running the notebook:

```bash
pip install prophet pmdarima xgboost scikit-learn pandas numpy matplotlib seaborn statsmodels
```

> **Note:** If running in Google Colab, the notebook installs these automatically in Step 1.

---

##  Dataset

**File:** `household_power_consumption.csv`  
**Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)

- ~2 million rows at 1-minute intervals
- Date range: December 2006 – November 2010
- Missing values are encoded as `?` — handled automatically during loading
- **Target variable:** `Global_active_power` (kilowatts)

You can upload the `.zip` or extracted `.csv` directly when prompted in the notebook.

---

##  Getting Started

### Option A — Google Colab (Recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run **Step 1** to install dependencies
3. Run **Step 3** — a file picker will appear; upload `household_power_consumption_csv.zip` or `household_power_consumption.csv`
4. Run all remaining cells in order

### Option B — Local Jupyter

```bash
# Clone or download the notebook
jupyter notebook energy_forecasting.ipynb
```

Make sure the dataset CSV is in the same directory as the notebook, then run all cells.

---

##  Models

### 1. ARIMA
- Uses `pmdarima.auto_arima` for automatic (p, d, q) order selection
- Trained on **daily** aggregated data
- Produces 95% confidence intervals
- Test period: **last 30 days**

### 2. Facebook Prophet
- Handles yearly and weekly seasonality automatically
- Multiplicative seasonality mode
- Includes uncertainty intervals
- Test period: **last 30 days**

### 3. XGBoost
- Treats time series as a **supervised regression** problem
- Uses engineered features: hour, day of week, month, lag values, rolling statistics, cyclical encodings
- 500 estimators with early stopping
- Test period: **last 7 days (168 hours, hourly)**

---

##  Feature Engineering

The following features are created for the XGBoost model:

| Category | Features |
|----------|----------|
| Calendar | `hour`, `dayofweek`, `dayofmonth`, `dayofyear`, `weekofyear`, `month`, `quarter`, `year` |
| Binary flags | `is_weekend`, `is_month_start`, `is_month_end` |
| Cyclical (sin/cos) | `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos` |
| Lag features | Previous 1, 2, 3, 6, 12, 24, 48 hours |
| Rolling stats | 24h and 168h rolling mean and standard deviation |

---

##  Evaluation Metrics

Models are evaluated on three standard regression metrics:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (kW) |
| **RMSE** | Root Mean Squared Error (kW) |
| **R²** | Coefficient of Determination |

---

##  Key Insights

1. **Peak energy usage** occurs between **18:00–21:00** (evening hours)
2. **Weekend consumption** differs from weekdays — later morning peak, different evening profile
3. **Winter months** show significantly higher consumption than summer
4. **XGBoost** benefits most from rich time-based and lag features for short-term hourly forecasting
5. **Prophet** captures seasonal patterns automatically — well-suited for weekly/monthly forecasting
6. **ARIMA** serves as a strong statistical baseline for near-stationary series

---

##  Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Short-term hourly forecasting | **XGBoost** (best with feature engineering) |
| Weekly / monthly forecasting | **Prophet** (handles seasonality automatically) |
| Quick statistical baseline | **ARIMA** (interpretable, minimal setup) |

---


##  Acknowledgements

- Dataset: [Georges Hébrail & Alice Bérard, UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
- Libraries: [Prophet](https://facebook.github.io/prophet/), [pmdarima](http://alkaline-ml.com/pmdarima/), [XGBoost](https://xgboost.readthedocs.io/), [statsmodels](https://www.statsmodels.org/)
