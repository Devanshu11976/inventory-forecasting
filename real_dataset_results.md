# Real Dataset Forecasting & Inventory Results (All 1,115 Stores)

This report compiles the walk-forward evaluation results, statistical significance tests, and working capital savings obtained by running the forecasting system on the **entire real Rossmann Store Sales dataset** (all 1,115 stores) over a 730-day historical horizon.

---

## 1. Dataset Metadata
* **Source**: Rossmann Store Sales (Kaggle)
* **Stores Used**: All 1,115 stores (representing StoreTypes A, B, C, and D)
* **Dataset Size**: 813,950 total daily sales records loaded into SQLite
* **Train / Test Split**: 670 training days / 60 testing days per store
* **Historical Horizon**: 730 days per store (August 1, 2013 to July 31, 2015)
* **Zero-Sales Handling**: Days with `Open = 0` (store closures) appear as `units_sold = 0`. MAPE calculation excludes them to prevent division-by-zero.

---

## 2. Overall Performance Metrics (Average Across All 1,115 Stores)

The table below shows the overall forecasting errors averaged across all 1,115 stores:

| Model | MAE | RMSE | MAPE % | sMAPE % |
| :--- | :--- | :--- | :---: | :---: |
| **ARIMA (2,1,2)** | 2232.67 | 2981.35 | 21.99% | 48.99% |
| **Baseline MA (7-day)** | 2209.85 | 3036.87 | 21.72% | 48.36% |
| **XGBoost (Default)** | **663.05** | **1148.99** | **9.35%** | **35.62%** |

---

## 3. Headline Improvements (XGBoost vs. Baselines)
* **XGBoost vs. Baseline MA**: **+70.0%** MAE improvement | **+57.0%** MAPE improvement
* **ARIMA vs. Baseline MA**: **-1.0%** MAE change (ARIMA underperforms baseline due to non-stationarity and store closures)
* **XGBoost vs. ARIMA**: **+70.3%** MAE improvement

---

## 4. Per-Store Breakdown (Representative Sample)

Below is the accuracy breakdown for a sample of Representative Stores (1, 4, 9 representing StoreTypes C, C, A):

* **Store 1** (StoreType C, Unit Cost = $20.00):
  * Baseline MA: MAE = `1243.86` / MAPE = `16.42%`
  * ARIMA: MAE = `1279.17` / MAPE = `17.16%`
  * XGBoost: **MAE = 360.53 / MAPE = 7.31%**
* **Store 4** (StoreType C, Unit Cost = $20.00):
  * Baseline MA: MAE = `2598.82` / MAPE = `15.88%`
  * ARIMA: MAE = `2756.33` / MAPE = `18.12%`
  * XGBoost: **MAE = 606.66 / MAPE = 6.80%**
* **Store 9** (StoreType A, Unit Cost = $10.00):
  * Baseline MA: MAE = `2255.64` / MAPE = `17.28%`
  * ARIMA: MAE = `2343.55` / MAPE = `18.74%`
  * XGBoost: **MAE = 826.50 / MAPE = 10.04%**

---

## 5. Statistical Significance (XGBoost vs. Baseline MA)

Paired $t$-tests and Diebold-Mariano tests ($h=1$) confirm that XGBoost's improvements are statistically significant at $p < 0.01$ across all stores. Example statistics for the representative sample:

| Store | StoreType | $t$-statistic | $t$-test $p$-value | DM-statistic | DM $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Store 1** | store_type_c | -5.3728 | 0.0000 | -4.1090 | 0.0000 |
| **Store 4** | store_type_c | -5.4403 | 0.0000 | -4.1614 | 0.0000 |
| **Store 9** | store_type_a | -4.9273 | 0.0000 | -4.2400 | 0.0000 |

---

## 6. Safety Stock & Working Capital Impact

By reducing the RMSE of the predictions, XGBoost allows businesses to hold significantly less safety stock (at a ~95% Cycle Service Level, $z=1.65$):

* **Store 1** (Unit Cost = $20):
  * Baseline Safety Stock: `6500.9` units | XGBoost Safety Stock: `2765.4` units
  * **Capital Released: $74,710**
* **Store 4** (Unit Cost = $20):
  * Baseline Safety Stock: `13869.1` units | XGBoost Safety Stock: `3595.3` units
  * **Capital Released: $205,477**
* **Store 9** (Unit Cost = $10):
  * Baseline Safety Stock: `12069.2` units | XGBoost Safety Stock: `5486.8` units
  * **Capital Released: $65,824**

### Total Working Capital Savings
* **Average Safety Stock Reduction**: **-62.0%** vs. Baseline MA
* **Total Estimated Capital Released across all 1,115 stores**: **$123,882,646.32**

---

## 7. Open-Days-Only Evaluation (Excluding Closure Days)

To assess if XGBoost's advantage holds up when dropping closure days (days where sales are zero because `Open = 0`), we evaluate metrics strictly on days the stores were trading:

### Aggregate Average (Open Days vs. All Days)

| Model | MAE (All Days) | MAE (Open Only) | sMAPE % (All Days) | sMAPE % (Open Only) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline MA** | 2209.85 | 1341.4 | 48.36% | 18.92% |
| **ARIMA** | 2232.67 | 1481.5 | 48.99% | 20.95% |
| **XGBoost** | **663.05** | **612.9** | **35.62%** | **8.21%** |

### Open-Day Insight
* **MAE Improvement on Open Days**: **+54.3%**
* **MAPE Improvement on Open Days**: **+51.3%**
* **Interpretation**: While calendar features successfully model closure days (predicting 0), XGBoost's forecasting advantage on normal trading days remains robust and statistically significant.
