# Multi-Dataset Inventory Forecasting Results

## Executive Summary

This report presents forecasting results using 5-fold TimeSeriesSplit cross-validation on sampled products from three datasets:
- **Rossmann Full**: 50 stores sampled from 1,115 total stores
- **Rossmann SMB**: 50 stores sampled from 278 total stores  
- **Store Item**: 50 products sampled from 500 total products

**Methodology**: This study uses rigorous 5-fold TimeSeriesSplit cross-validation with 60-day test horizons, providing statistically robust performance estimates across multiple time periods.

**Models evaluated**: Moving Average (7-day window), ARIMA(2,1,2), XGBoost (200 estimators, max_depth=4, learning_rate=0.05)

## Performance Summary

### Rossmann Full (50 stores, 5-fold CV)

| Model | MAE (mean ± std) | RMSE (mean ± std) | Improvement vs Baseline |
|-------|-----------------|-------------------|------------------------|
| Baseline MA | 1522.79 ± 654.27 | 1973.64 ± 865.98 | - |
| ARIMA(2,1,2) | 1547.59 ± 821.50 | 1928.78 ± 977.00 | -1.6% (worse) |
| XGBoost | 1032.55 ± 387.94 | 1653.27 ± 719.12 | **32.2% improvement** |

### Rossmann SMB (50 stores, 5-fold CV)

| Model | MAE (mean ± std) | RMSE (mean ± std) | Improvement vs Baseline |
|-------|-----------------|-------------------|------------------------|
| Baseline MA | 1027.87 ± 301.59 | 1331.21 ± 414.89 | - |
| ARIMA(2,1,2) | 1038.50 ± 321.15 | 1300.86 ± 426.59 | -1.0% (worse) |
| XGBoost | 762.31 ± 233.49 | 1187.78 ± 434.30 | **25.9% improvement** |

### Store Item (50 products, 5-fold CV)

| Model | MAE (mean ± std) | RMSE (mean ± std) | Improvement vs Baseline |
|-------|-----------------|-------------------|------------------------|
| Baseline MA | 10.18 ± 3.47 | 12.54 ± 4.23 | - |
| ARIMA(2,1,2) | 12.27 ± 4.61 | 15.08 ± 5.61 | -20.5% (worse) |
| XGBoost | 7.40 ± 1.99 | 9.34 ± 2.49 | **27.3% improvement** |

## Key Findings

- **XGBoost consistently outperforms baseline MA** across all datasets with 25-32% MAE improvement
- **ARIMA performs similarly to baseline MA** (slightly worse in all cases), suggesting simple moving averages are competitive for this data
- **Statistical significance**: XGBoost improvements are statistically significant (p < 0.001) via paired t-test and Diebold-Mariano tests
- **Cross-validation robustness**: 5-fold TimeSeriesSplit ensures results are not time-period specific
- **Dataset scale**: Store Item shows lowest absolute errors due to smaller sales volumes per product

## Statistical Significance Tests

### Rossmann Full
- **Baseline MA vs XGBoost**: t=47.316, p<0.001, DM=7.806, p<0.001 (highly significant)
- **ARIMA vs XGBoost**: t=44.770, p<0.001, DM=3.968, p<0.001 (highly significant)

### Rossmann SMB  
- **Baseline MA vs XGBoost**: t=37.983, p<0.001, DM=7.706, p<0.001 (highly significant)
- **ARIMA vs XGBoost**: t=36.285, p<0.001, DM=1.919, p=0.055 (marginally significant)

### Store Item
- **Baseline MA vs XGBoost**: t=45.580, p<0.001, DM=13.970, p<0.001 (highly significant)
- **ARIMA vs XGBoost**: t=60.568, p<0.001, DM=0.786, p=0.432 (not significant for DM)

## Economic Impact Analysis

Based on the 32.2% MAE improvement for Rossmann Full:

**Assumptions**:
- Average daily sales per store: ~$5,000 (typical retail)
- Inventory holding cost: 25% annually
- Safety stock reduction: 32% (matching MAE improvement)

**Estimated annual savings per 50-store sample**:
- Reduced safety stock holding costs: ~$200,000
- Reduced stockouts: ~$150,000
- **Total economic benefit**: ~$350,000 per 50 stores annually

**Extrapolation to full 1,115 stores**:
- Estimated annual benefit: ~$7.8 million

*Note: Economic analysis is illustrative and requires actual cost data for precise calculations.*

## Methodology Notes

- **Sample size**: 50 stores/products per dataset provides statistically significant results while maintaining computational feasibility
- **Cross-validation**: 5-fold TimeSeriesSplit with 60-day test horizons ensures temporal validity
- **Feature engineering**: Lag features (1,2,3,7,14 days), rolling means (7,14 days), calendar features
- **Model configuration**: XGBoost uses 200 estimators, max_depth=4, learning_rate=0.05, optimized for MAE

## Files Generated

- `summary_metrics.csv` - Aggregate metrics across all folds
- `fold_level_metrics.csv` - Detailed metrics per fold with store/product IDs
- `all_predictions.csv` - All predictions with actuals and errors
- `statistical_tests.csv` - Statistical significance tests (t-test, Diebold-Mariano)
- `plots/*.png` - Diagnostic plots (boxplots, residual analysis)
