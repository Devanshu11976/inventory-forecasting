import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag, rolling-window, and calendar features.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least:
            - 'product_id' (str)
            - 'date' (str or datetime)
            - 'units_sold' (int/float)
            
    Returns:
        pd.DataFrame: A new DataFrame with the added features.
                      The original input DataFrame is left unmodified.
    """
    # Create copy to prevent mutating raw history (Section III-A)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
    
    # Lags (Section III-C: trained on lags {1,2,3,7,14})
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df.groupby('product_id')['units_sold'].shift(lag)
        
    # Rolling window means (Section III-C: 7- and 14-day rolling means)
    # We shift by 1 day before rolling to prevent data leakage (since we cannot see today's actual sales at prediction time)
    df['rolling_mean_7'] = df.groupby('product_id')['units_sold'].shift(1).rolling(window=7).mean()
    df['rolling_mean_14'] = df.groupby('product_id')['units_sold'].shift(1).rolling(window=14).mean()
    
    # Calendar features (Section III-C: day-of-week, day-of-year)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    
    return df
