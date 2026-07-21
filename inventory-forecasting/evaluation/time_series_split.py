import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesCrossValidation:
    """
    TimeSeriesSplit with 5 folds for time series cross-validation.
    Ensures chronological order and prevents data leakage.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 60):
        """
        Args:
            n_splits: Number of folds (default: 5)
            test_size: Number of days in test set per fold (default: 60)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
    def split(self, df: pd.DataFrame, product_id: str) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data for a single product into train/test folds.
        
        Args:
            df: DataFrame for a single product with columns: product_id, date, units_sold
            product_id: Product identifier
            
        Returns:
            List of (train_df, test_df) tuples for each fold
        """
        df = df.sort_values('date').reset_index(drop=True)
        
        # Need minimum data for 5 folds with 60-day test size
        min_required = self.n_splits * self.test_size + 100  # +100 for training buffer
        if len(df) < min_required:
            print(f"  Warning: Product {product_id} has only {len(df)} rows, "
                  f"less than recommended {min_required}. Using single split.")
            # Use single split if insufficient data
            split_point = max(100, len(df) - self.test_size)
            return [(df.iloc[:split_point], df.iloc[split_point:])]
        
        folds = []
        for train_idx, test_idx in self.tscv.split(df):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            folds.append((train_df, test_df))
            
        return folds
    
    def get_fold_info(self, df: pd.DataFrame) -> List[dict]:
        """
        Get information about each fold (train size, test size, date ranges).
        """
        df = df.sort_values('date').reset_index(drop=True)
        fold_info = []
        
        for i, (train_idx, test_idx) in enumerate(self.tscv.split(df)):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            fold_info.append({
                'fold': i + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_date_start': train_df['date'].min(),
                'train_date_end': train_df['date'].max(),
                'test_date_start': test_df['date'].min(),
                'test_date_end': test_df['date'].max()
            })
            
        return fold_info


if __name__ == "__main__":
    # Test the TimeSeriesSplit
    import sys
    sys.path.append('c:/Users/devan/Desktop/research paper/inventory-forecasting')
    from datasets.multi_dataset_loader import load_all_datasets
    
    datasets, info = load_all_datasets('c:/Users/devan/Desktop/research paper/dataset')
    
    tscv = TimeSeriesCrossValidation(n_splits=5, test_size=60)
    
    # Test on first product of each dataset
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name}:")
        first_product = df['product_id'].iloc[0]
        product_df = df[df['product_id'] == first_product].copy()
        
        folds = tscv.split(product_df, first_product)
        print(f"  Product {first_product}: {len(folds)} folds")
        
        for i, (train, test) in enumerate(folds):
            print(f"    Fold {i+1}: Train {len(train)} rows ({train['date'].min()} to {train['date'].max()}), "
                  f"Test {len(test)} rows ({test['date'].min()} to {test['date'].max()})")
