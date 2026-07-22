import pandas as pd
import numpy as np
from models.baseline_ma import BaselineMA
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel

def run_walk_forward_evaluation(df_sales: pd.DataFrame, model_name: str, tune: bool = False) -> pd.DataFrame:
    """
    Evaluates a model using the strict chronological split specified in the paper:
    - 670 days training
    - 60 days testing
    
    Args:
        df_sales (pd.DataFrame): Sales history DataFrame for a single product with:
            - 'product_id' (str)
            - 'date' (datetime)
            - 'units_sold' (int/float)
        model_name (str): One of ['baseline_ma', 'arima', 'xgboost']
        tune (bool): If True, enables hyperparameter search for ARIMA and XGBoost.
        
    Returns:
        pd.DataFrame: DataFrame containing product_id, date, model_name, predicted_units, actual_units.
    """
    # Sort history chronologically
    df_sales = df_sales.sort_values('date').reset_index(drop=True)
    
    total_days = len(df_sales)
    test_days = 60
    train_days = total_days - test_days
    
    # Chronological split (Section III-D: last 60 days test, preceding 670 days train)
    train_df = df_sales.iloc[:train_days].copy()
    test_df = df_sales.iloc[train_days:].copy()
    
    test_dates = test_df['date'].dt.strftime('%Y-%m-%d').values
    test_actuals = test_df['units_sold'].values
    
    predictions = []
    
    # Algorithm 1 implementation:
    if model_name == 'baseline_ma':
        # Re-fit/update per step for baseline (moving average)
        model = BaselineMA(window_size=7)
        model.fit(train_df)
        for i in range(test_days):
            pred = model.predict(1)[0]
            predictions.append(pred)
            model.update_history(test_actuals[i])
            
    elif model_name == 'arima':
        # Fit once on training, forecast full test horizon directly
        model = ARIMAModel(auto_arima=tune)
        model.fit(train_df)
        predictions = model.predict(test_days)
        
    elif model_name == 'xgboost':
        # Fit once on training, walk-forward prediction with actuals updating features
        model = XGBoostModel(tune=tune)
        model.fit(train_df)
        for i in range(test_days):
            pred = model.predict(1)[0]
            predictions.append(pred)
            model.update_history(test_dates[i], test_actuals[i])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    results_df = pd.DataFrame({
        'product_id': df_sales['product_id'].iloc[0],
        'date': test_dates,
        'model_name': model_name,
        'predicted_units': predictions,
        'actual_units': test_actuals
    })
    
    return results_df
