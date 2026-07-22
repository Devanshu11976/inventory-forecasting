# AI-Based Inventory Forecasting for Small Businesses

This repository contains the complete, production-quality Python implementation of the forecasting system described in the paper **"AI-Based Inventory Forecasting for Small Businesses"**.

---

## 1. Project Structure

```
inventory-forecasting/
├── data/
│   ├── schema.sql              # SQLite schema: products, sales_history, forecasts
│   ├── generator.py            # Synthetic data generator (Eq. 1)
│   ├── table_ii.csv            # Output performance table (CSV)
│   └── table_iii.csv           # Output category breakdown table (CSV)
├── features/
│   └── engineering.py          # Lag, rolling-window, and calendar features
├── models/
│   ├── base.py                 # Common model interface (fit/predict)
│   ├── baseline_ma.py          # 7-day moving average baseline model
│   ├── arima_model.py          # ARIMA(2,1,2) + auto_arima grid search
│   └── xgboost_model.py        # XGBoost Regressor + parameter grid search
├── evaluation/
│   ├── metrics.py              # Mathematical formulas for MAE, RMSE, MAPE, sMAPE
│   ├── walk_forward.py         # Algorithm 1: Walk-forward validation
│   └── significance.py         # Paired t-test and Diebold-Mariano tests
├── decision/
│   └── reorder_point.py        # Reorder point & safety stock formulas (Eq. Section VII)
├── datasets/
│   └── real_data_loader.py     # Ingestion hook for Rossmann Store Sales
├── tests/
│   ├── test_metrics.py         # Unit tests for forecasting metrics
│   └── test_reorder.py         # Unit tests for inventory decision rules
├── run_experiment.py           # Unified entry point orchestrator
└── requirements.txt            # Pinned dependencies
```

---

## 2. Requirements & Installation

Make sure Python 3.12+ is installed. To install all pinned dependencies, run:

```bash
pip install -r inventory-forecasting/requirements.txt
```

---

## 3. Dataset Setup

This project uses the **Rossmann Store Sales** dataset, publicly available on Kaggle:

> 📦 **Download here:** [https://www.kaggle.com/c/rossmann-store-sales/data](https://www.kaggle.com/c/rossmann-store-sales/data)

After downloading, place the files in a `dataset/` folder **one level above** this repository (i.e., alongside `inventory-forecasting/`):

```
research paper/
├── dataset/
│   ├── train.csv      ← main sales data (1,115 stores, 730 days)
│   ├── store.csv      ← store metadata (type, assortment, etc.)
│   └── test.csv       ← optional
└── inventory-forecasting/
    └── ...
```

> **Note:** The `dataset/` CSV files are **not tracked by git** (large files). You must download them from Kaggle before running any experiments.

---

## 4. How to Run the System

### Run the Default Experiment (Real Data: Rossmann Store Sales)
To run the forecasting models on the real Rossmann Store Sales dataset located in the `dataset/train.csv` and `dataset/store.csv` folders outside this directory, simply run:

```bash
$env:PYTHONPATH="inventory-forecasting"; python inventory-forecasting/run_experiment.py --store-ids 1 2 3
```

This loads stores 1, 2, and 3, initializes/clears the SQLite database at `inventory-forecasting/data/inventory.db`, and runs walk-forward evaluation, models, significance tests, and reorder point calculations on the real data.

### Run on Synthetic Data
To reproduce the synthetic evaluation tables (Table II and Table III) and save the visualization of the grocery staple archetype forecast (Fig. 2), use the `--synthetic-data` flag:

```bash
$env:PYTHONPATH="inventory-forecasting"; python inventory-forecasting/run_experiment.py --synthetic-data --seed 42
```

This will:
1. Initialize/clear the SQLite database at `inventory-forecasting/data/inventory.db`.
2. Generate synthetic demand for the three archetypes.
3. Fit all models and execute walk-forward evaluation.
4. Calculate metrics, run t-tests/DM-tests, and output Markdown tables to stdout.
5. Save the generated tables in CSV format and save the grocery staple forecast chart as `inventory-forecasting/grocery_staple_forecast.png`.
6. Reproduce the worked safety stock reduction worked calculation ($3,000 savings).

### Run with Hyperparameter Tuning
To enable ARIMA order search and XGBoost grid search parameter tuning, add the `--tune` flag:

```bash
$env:PYTHONPATH="inventory-forecasting"; python inventory-forecasting/run_experiment.py --tune
```

---

## 5. Verification & Testing

To run the full suite of unit tests verifying metrics and inventory calculations under edge cases (empty data, division by zero, etc.), run:

```bash
$env:PYTHONPATH="inventory-forecasting"; python -m pytest inventory-forecasting/tests/
```
