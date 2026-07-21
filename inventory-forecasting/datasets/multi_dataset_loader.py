import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import sqlite3

class MultiDatasetLoader:
    """
    Unified loader for 3 dataset conditions:
    1. Full Rossmann (all 1,115 stores)
    2. Rossmann SMB-proxy subset (bottom-quartile stores)
    3. Store Item Demand Forecasting dataset (10 stores, 50 items)
    
    Uses identical feature engineering and model configs across all conditions.
    """
    
    def __init__(self):
        self.dataset_info = {}
        
    def load_rossmann_full(self, train_csv_path: str, store_csv_path: str) -> pd.DataFrame:
        """
        Load full Rossmann dataset (all 1,115 stores).
        Returns unified DataFrame with columns: product_id, date, units_sold
        """
        print("Loading Full Rossmann dataset...")
        df_train = pd.read_csv(train_csv_path, low_memory=False)
        df_store = pd.read_csv(store_csv_path)
        
        # Map to unified schema
        df_train['product_id'] = df_train['Store'].astype(str)
        df_train['date'] = pd.to_datetime(df_train['Date'])
        df_train['units_sold'] = df_train['Sales']
        
        # Filter only open stores (Sales > 0 or Open == 1)
        df_train = df_train[(df_train['Open'] == 1) | (df_train['Sales'] > 0)].copy()
        
        # Select relevant columns
        df_unified = df_train[['product_id', 'date', 'units_sold']].copy()
        df_unified = df_unified.sort_values(['product_id', 'date']).reset_index(drop=True)
        
        self.dataset_info['rossmann_full'] = {
            'rows': len(df_unified),
            'products': df_unified['product_id'].nunique(),
            'date_range': (df_unified['date'].min(), df_unified['date'].max()),
            'source': 'Rossmann Store Sales (Kaggle)',
            'description': 'Full dataset with all 1,115 stores'
        }
        
        print(f"  Loaded {len(df_unified)} rows for {df_unified['product_id'].nunique()} stores")
        print(f"  Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
        
        return df_unified
    
    def load_rossmann_smb(self, train_csv_path: str, store_csv_path: str) -> pd.DataFrame:
        """
        Load Rossmann SMB-proxy subset (bottom-quartile stores by volume).
        Returns unified DataFrame with columns: product_id, date, units_sold
        """
        print("Loading Rossmann SMB-proxy subset...")
        df_train = pd.read_csv(train_csv_path, low_memory=False)
        df_store = pd.read_csv(store_csv_path)
        
        # Map to unified schema
        df_train['product_id'] = df_train['Store'].astype(str)
        df_train['date'] = pd.to_datetime(df_train['Date'])
        df_train['units_sold'] = df_train['Sales']
        
        # Filter only open stores
        df_train = df_train[(df_train['Open'] == 1) | (df_train['Sales'] > 0)].copy()
        
        # Select relevant columns
        df_unified = df_train[['product_id', 'date', 'units_sold']].copy()
        df_unified = df_unified.sort_values(['product_id', 'date']).reset_index(drop=True)
        
        self.dataset_info['rossmann_smb'] = {
            'rows': len(df_unified),
            'products': df_unified['product_id'].nunique(),
            'date_range': (df_unified['date'].min(), df_unified['date'].max()),
            'source': 'Rossmann Store Sales (Kaggle) - SMB subset',
            'description': 'Bottom quartile stores by average daily sales (278 stores)'
        }
        
        print(f"  Loaded {len(df_unified)} rows for {df_unified['product_id'].nunique()} stores")
        print(f"  Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
        
        return df_unified
    
    def load_store_item(self, train_csv_path: str) -> pd.DataFrame:
        """
        Load Store Item Demand Forecasting dataset.
        Returns unified DataFrame with columns: product_id, date, units_sold
        """
        print("Loading Store Item Demand Forecasting dataset...")
        df_train = pd.read_csv(train_csv_path)
        
        # Create composite product_id: store_item format
        df_train['product_id'] = df_train['store'].astype(str) + '_' + df_train['item'].astype(str)
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_train['units_sold'] = df_train['sales']
        
        # Select relevant columns
        df_unified = df_train[['product_id', 'date', 'units_sold']].copy()
        df_unified = df_unified.sort_values(['product_id', 'date']).reset_index(drop=True)
        
        self.dataset_info['store_item'] = {
            'rows': len(df_unified),
            'products': df_unified['product_id'].nunique(),
            'date_range': (df_unified['date'].min(), df_unified['date'].max()),
            'source': 'Store Item Demand Forecasting Challenge (Kaggle)',
            'description': '10 stores × 50 items = 500 product combinations'
        }
        
        print(f"  Loaded {len(df_unified)} rows for {df_unified['product_id'].nunique()} product combinations")
        print(f"  Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
        
        return df_unified
    
    def get_dataset_info(self) -> Dict:
        """Return summary information about all loaded datasets."""
        return self.dataset_info
    
    def print_dataset_summary(self):
        """Print formatted summary of all datasets."""
        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"{'Dataset':<20} {'Rows':>12} {'Products':>10} {'Date Range':>30}")
        print("-" * 80)
        
        for name, info in self.dataset_info.items():
            date_str = f"{info['date_range'][0].strftime('%Y-%m-%d')} to {info['date_range'][1].strftime('%Y-%m-%d')}"
            print(f"{name:<20} {info['rows']:>12,} {info['products']:>10} {date_str:>30}")
        
        print("=" * 80)


def load_all_datasets(base_path: str) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all 3 datasets at once.
    
    Args:
        base_path: Base path to dataset directory
        
    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    loader = MultiDatasetLoader()
    
    datasets = {}
    
    # Full Rossmann
    datasets['rossmann_full'] = loader.load_rossmann_full(
        f'{base_path}/train.csv',
        f'{base_path}/store.csv'
    )
    
    # Rossmann SMB subset
    datasets['rossmann_smb'] = loader.load_rossmann_smb(
        f'{base_path}/rossmann_smb_train.csv',
        f'{base_path}/rossmann_smb_store.csv'
    )
    
    # Store Item Demand
    datasets['store_item'] = loader.load_store_item(
        f'{base_path}/store_item_train.csv'
    )
    
    loader.print_dataset_summary()
    
    return datasets, loader.get_dataset_info()


if __name__ == "__main__":
    # Test the loader
    base_path = 'c:/Users/devan/Desktop/research paper/dataset'
    datasets, info = load_all_datasets(base_path)
    
    print("\nDataset info dictionary:")
    for name, data in info.items():
        print(f"{name}: {data}")
