import numpy as np

def mae(actual, predicted) -> float:
    """
    Mean Absolute Error (Eq. 2).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    if len(actual) == 0:
        return 0.0
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual, predicted) -> float:
    """
    Root Mean Squared Error (Eq. 3).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    if len(actual) == 0:
        return 0.0
    return float(np.sqrt(np.mean((actual - predicted)**2)))

def mape(actual, predicted) -> float:
    """
    Mean Absolute Percentage Error (Eq. 4), excluding zero-actual days to avoid division by zero.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    if len(actual) == 0:
        return 0.0
    
    # Exclude zero-actuals (Section III-D: exclude zero-actual days)
    mask = actual != 0
    if not np.any(mask):
        return 0.0
        
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)

def smape(actual, predicted) -> float:
    """
    Symmetric Mean Absolute Percentage Error (Eq. 5).
    Safely handles cases where both actual and predicted are zero.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    if len(actual) == 0:
        return 0.0
        
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    
    # Exclude cases where both actual and predicted are 0 to avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
        
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100.0)
