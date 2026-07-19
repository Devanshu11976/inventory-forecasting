import numpy as np
import scipy.stats as stats

def paired_t_test(actual, pred_xgboost, pred_baseline) -> tuple:
    """
    Computes a paired t-test on the daily absolute errors between XGBoost and Baseline MA.
    
    Args:
        actual (array-like): True actual values.
        pred_xgboost (array-like): XGBoost predictions.
        pred_baseline (array-like): Baseline MA predictions.
        
    Returns:
        tuple: (t_statistic, p_value)
    """
    actual = np.array(actual)
    p_xgb = np.array(pred_xgboost)
    p_base = np.array(pred_baseline)
    
    ae_xgb = np.abs(actual - p_xgb)
    ae_base = np.abs(actual - p_base)
    
    stat, pval = stats.ttest_rel(ae_xgb, ae_base)
    return float(stat), float(pval)

def diebold_mariano_test(actual, pred_xgboost, pred_baseline, h=1) -> tuple:
    """
    Computes the Diebold-Mariano test statistic for squared error loss.
    Uses the standard normal distribution under the null hypothesis of equal forecast accuracy.
    
    Args:
        actual (array-like): True actual values.
        pred_xgboost (array-like): XGBoost predictions.
        pred_baseline (array-like): Baseline MA predictions.
        h (int): Forecast horizon (number of steps ahead), defaults to 1.
        
    Returns:
        tuple: (dm_statistic, p_value)
    """
    actual = np.array(actual)
    p1 = np.array(pred_xgboost)
    p2 = np.array(pred_baseline)
    
    e1 = actual - p1
    e2 = actual - p2
    
    # Loss differential using squared-error loss (Section V-C: loss differential / squared-error loss)
    d = e1**2 - e2**2
    n = len(d)
    
    d_bar = np.mean(d)
    
    # Compute autocovariances up to lag h-1 to account for autocorrelation in multi-step forecasts
    gamma = np.zeros(h)
    for k in range(h):
        if k == 0:
            gamma[k] = np.var(d, ddof=0)
        else:
            gamma[k] = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            
    # Estimate the variance of the sample mean d_bar
    var_d = gamma[0] + 2.0 * np.sum(gamma[1:])
    
    if var_d <= 0:
        dm_stat = 0.0
        p_value = 1.0
    else:
        # Standard normal test statistic
        dm_stat = d_bar / np.sqrt(var_d / n)
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
        
    return float(dm_stat), float(p_value)
