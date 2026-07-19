import pandas as pd

class BaseModel:
    """
    Common interface for all forecasting models.
    """
    def fit(self, train_df: pd.DataFrame):
        """
        Fits the model on the training data.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame containing 'date' and 'units_sold' columns.
        """
        raise NotImplementedError
        
    def predict(self, n_days: int) -> list:
        """
        Generates predictions for the next n_days.
        
        Args:
            n_days (int): Number of days to forecast.
            
        Returns:
            list: Predicted sales values.
        """
        raise NotImplementedError
