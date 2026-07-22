import numpy as np
import pandas as pd

def calculate_reorder_point(lead_time_predictions, rmse, lead_time, z=1.65) -> tuple:
    """
    Computes the reorder point and safety stock.
    
    reorder_point = y_hat_lead_time + z * RMSE * sqrt(lead_time) (Section VII)
    
    Args:
        lead_time_predictions (array-like): Daily forecast values for the lead time period.
        rmse (float): RMSE of the forecasting model on validation/test set.
        lead_time (int): Configurable lead time (in days).
        z (float): Service-level multiplier (default 1.65 for ~95% service level).
        
    Returns:
        tuple: (forecasted_demand_during_lead_time, safety_stock, reorder_point)
    """
    # ŷ_lead_time is the sum of forecasted demand over the lead time period
    forecasted_demand = float(np.sum(lead_time_predictions[:lead_time]))
    
    # Safety Stock = z * RMSE * sqrt(lead_time)
    safety_stock = float(z * rmse * np.sqrt(lead_time))
    
    reorder_point = forecasted_demand + safety_stock
    
    return forecasted_demand, safety_stock, reorder_point

def run_worked_example() -> dict:
    """
    Reproduces the worked $3,000-safety-stock-reduction example from Section V-A of the paper:
    
    Product: High value staple
    - Unit Cost: $50.00
    - Lead Time: 16 days
    - Service Level: 95% (z = 1.65)
    - Baseline MA RMSE: 20.20
    - XGBoost RMSE: 11.11
    
    Formula:
    Safety Stock (Baseline) = 1.65 * 20.20 * sqrt(16) = 133.32 units
    Safety Stock (XGBoost) = 1.65 * 11.11 * sqrt(16) = 73.326 units
    Reduction = 133.32 - 73.326 = 59.994 units
    Savings = 59.994 * $50.00 = $2,999.70 (~$3,000)
    """
    unit_cost = 50.00
    lead_time = 16
    z = 1.65
    rmse_baseline = 20.20
    rmse_xgboost = 11.11
    
    ss_baseline = z * rmse_baseline * np.sqrt(lead_time)
    ss_xgboost = z * rmse_xgboost * np.sqrt(lead_time)
    
    reduction_units = ss_baseline - ss_xgboost
    savings_dollars = reduction_units * unit_cost
    
    return {
        'unit_cost': unit_cost,
        'lead_time': lead_time,
        'z': z,
        'rmse_baseline': rmse_baseline,
        'rmse_xgboost': rmse_xgboost,
        'ss_baseline_units': ss_baseline,
        'ss_xgboost_units': ss_xgboost,
        'reduction_units': reduction_units,
        'savings_dollars': savings_dollars
    }
