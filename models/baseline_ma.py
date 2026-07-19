import pandas as pd
import numpy as np
from models.base import BaseModel

class BaselineMA(BaseModel):
    """
    Baseline model: 7-day moving average.
    """
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.history = []
        
    def fit(self, train_df: pd.DataFrame):
        """
        Saves the training series as the initial history window.
        """
        self.history = list(train_df['units_sold'].values)
        
    def predict(self, n_days: int) -> list:
        """
        Forecasts n_days by recursively calculating the moving average.
        For one-step-ahead walk-forward evaluation, this is called with n_days=1.
        """
        predictions = []
        temp_history = list(self.history)
        
        for _ in range(n_days):
            if not temp_history:
                pred = 0.0
            else:
                # 7-day moving average (Section III-C: 7-day moving average)
                pred = np.mean(temp_history[-self.window_size:])
            predictions.append(pred)
            temp_history.append(pred)
            
        return predictions
        
    def update_history(self, actual_value: float):
        """
        Appends true actual sales to the history window at each walk-forward step.
        """
        self.history.append(actual_value)
