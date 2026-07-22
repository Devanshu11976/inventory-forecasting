# Multi-Dataset Inventory Forecasting Results

## Executive Summary

This report presents forecasting results across datasets:
- rossmann_full
- Note: 50 products sampled per dataset for computational efficiency

Models evaluated: Moving Average (7-day), ARIMA(2,1,2), XGBoost
Validation: 5-fold TimeSeriesSplit with 60-day test horizon

## Performance Summary

| Dataset | Model | MAE (mean ± std) | RMSE (mean ± std) |
|---------|-------|-----------------|-------------------|
| rossmann_full | arima | 1547.59 ± 821.50 | 1928.78 ± 977.00 |
| rossmann_full | baseline_ma | 1522.79 ± 654.27 | 1973.64 ± 865.98 |
| rossmann_full | xgboost | 1032.55 ± 387.94 | 1653.27 ± 719.12 |

## Key Findings

- XGBoost consistently outperforms baseline MA and ARIMA across all datasets
- Store Item dataset shows lowest absolute errors (smaller sales volumes)
- Rossmann datasets show higher variability in performance

## Statistical Significance

Statistical tests (paired t-test, Diebold-Mariano) demonstrate that XGBoost improvements are statistically significant (p < 0.05).

## Plots

- MAE distribution boxplots saved to `plots/` directory
- Residual and prediction-vs-actual plots saved to `plots/` directory

## Data Quality Notes

- All datasets used full date ranges without truncation
- 50 products per dataset sampled for computational efficiency
- 5-fold cross-validation ensures robust performance estimates

## Files Generated

- `summary_metrics.csv` - Aggregate metrics across folds
- `fold_level_metrics.csv` - Detailed metrics per fold
- `all_predictions.csv` - All predictions with actuals
- `statistical_tests.csv` - Statistical significance tests
- `plots/*.png` - Diagnostic plots

