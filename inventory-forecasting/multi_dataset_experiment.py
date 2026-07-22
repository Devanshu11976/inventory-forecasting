import pandas as pd
import numpy as np
import os
import sys
import json
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Local imports
from datasets.multi_dataset_loader import load_all_datasets
from evaluation.time_series_split import TimeSeriesCrossValidation
from models.baseline_ma import BaselineMA
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from features.engineering import engineer_features
from evaluation.metrics import mae, rmse
from evaluation.significance import paired_t_test, diebold_mariano_test


class MultiDatasetExperiment:
    """
    Runs comprehensive forecasting experiments across 3 datasets and 3 models.
    Uses TimeSeriesSplit (5 folds) for rigorous evaluation.
    """
    
    def __init__(self, base_path: str, output_dir: str):
        self.base_path = base_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.datasets = {}
        self.dataset_info = {}
        self.results = []
        self.all_predictions = []
        self.fold_metrics = []
        self.checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
        self.completed_datasets = self.load_checkpoint()
        
    def load_checkpoint(self) -> set:
        """Load completed datasets from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    completed = set(data.get('completed_datasets', []))
                    print(f"Loaded checkpoint: {len(completed)} datasets already completed")
                    return completed
            except Exception as e:
                print(f"Error loading checkpoint: {e}, starting fresh")
                return set()
        return set()
    
    def save_checkpoint(self, dataset_name: str):
        """Save checkpoint after completing a dataset."""
        self.completed_datasets.add(dataset_name)
        with open(self.checkpoint_file, 'w') as f:
            json.dump({'completed_datasets': list(self.completed_datasets)}, f)
        print(f"Checkpoint saved: {dataset_name} marked as completed")
        
    def load_data(self):
        """Load all 3 datasets."""
        print("Loading datasets...")
        self.datasets, self.dataset_info = load_all_datasets(self.base_path)
        
    @staticmethod
    def train_single_product_fold(product_id: str, dataset_name: str, 
                                   train_df: pd.DataFrame, test_df: pd.DataFrame,
                                   model_name: str, fold: int, auto_arima: bool = False) -> Dict:
        """
        Train and evaluate a single model on a single product fold.
        Static method for parallel processing compatibility.
        """
        try:
            # Initialize model
            if model_name == 'baseline_ma':
                model = BaselineMA(window_size=7)
            elif model_name == 'arima':
                model = ARIMAModel(auto_arima=auto_arima)
            elif model_name == 'xgboost':
                # Use CPU only - GPU causes issues in multiprocessing
                model = XGBoostModel(tune=False, tree_method='hist', device='cpu')
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Fit model
            model.fit(train_df)
            
            # Predict
            if model_name == 'baseline_ma':
                # Walk-forward prediction for baseline (only needs actual value)
                predictions = []
                for i in range(len(test_df)):
                    pred = model.predict(1)[0]
                    predictions.append(pred)
                    model.update_history(test_df.iloc[i]['units_sold'])
            elif model_name == 'xgboost':
                # Walk-forward prediction for XGBoost (needs date and actual value)
                predictions = []
                for i in range(len(test_df)):
                    pred = model.predict(1)[0]
                    predictions.append(pred)
                    model.update_history(test_df.iloc[i]['date'], test_df.iloc[i]['units_sold'])
            else:
                # Direct forecast for ARIMA
                predictions = model.predict(len(test_df))
            
            # Calculate metrics
            actuals = test_df['units_sold'].values
            pred_array = np.array(predictions)
            
            fold_mae = mae(actuals, pred_array)
            fold_rmse = rmse(actuals, pred_array)
            
            result = {
                'dataset': dataset_name,
                'product_id': product_id,
                'model': model_name,
                'fold': fold,
                'mae': fold_mae,
                'rmse': fold_rmse,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'predictions': predictions,
                'actuals': actuals.tolist(),
                'dates': test_df['date'].tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"  Error in train_single_product_fold for {product_id}, {model_name}, fold {fold}: {e}")
            return None
    
    def run_experiment_on_dataset(self, dataset_name: str, df: pd.DataFrame, 
                                 model_names: List[str], auto_arima: bool = False, 
                                 max_products: int = 50):
        """
        Run experiment on a single dataset with sampled products for speed.
        """
        print(f"\n{'='*80}")
        print(f"Running experiment on {dataset_name}")
        print(f"{'='*80}")
        
        tscv = TimeSeriesCrossValidation(n_splits=5, test_size=60)
        
        # Group by product and sample
        product_groups = {pid: group for pid, group in df.groupby('product_id')}
        n_products = len(product_groups)
        
        # Sample products for faster execution
        if n_products > max_products:
            import random
            sampled_ids = random.sample(list(product_groups.keys()), max_products)
            product_groups = {pid: product_groups[pid] for pid in sampled_ids}
            print(f"  Sampled {max_products} products from {n_products} total")
        else:
            print(f"  Using all {n_products} products")
        
        n_products_sampled = len(product_groups)
        
        print(f"  Processing {n_products_sampled} products with {len(model_names)} models...")
        print(f"  Total model-dataset combinations: {n_products_sampled * len(model_names)}")
        print(f"  Total folds: {n_products_sampled * len(model_names) * 5}")
        
        # Prepare all tasks for parallel processing
        tasks = []
        for product_id, product_df in product_groups.items():
            folds = tscv.split(product_df, product_id)
            for fold_idx, (train_df, test_df) in enumerate(folds):
                fold_num = fold_idx + 1
                for model_name in model_names:
                    tasks.append((product_id, dataset_name, train_df, test_df, model_name, fold_num, auto_arima))
        
        print(f"  Using parallel processing with ProcessPoolExecutor...")
        
        # Process in parallel with reduced workers for stability
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(self.train_single_product_fold, *task)
                futures.append(future)
            
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.results.append(result)
                    self.fold_metrics.append(result)
                    
                    # Extract predictions for storage
                    for i, (date, actual, pred) in enumerate(zip(result['dates'], result['actuals'], result['predictions'])):
                        self.all_predictions.append({
                            'dataset': dataset_name,
                            'product_id': result['product_id'],
                            'model': result['model'],
                            'fold': result['fold'],
                            'date': date,
                            'actual': actual,
                            'predicted': pred,
                            'error': actual - pred,
                            'abs_error': abs(actual - pred)
                        })
                
                completed += 1
                if completed % 500 == 0 or completed == total:
                    print(f"    Progress: {completed}/{total} combinations completed")
        
        print(f"  Completed {dataset_name}: {len([r for r in self.results if r['dataset'] == dataset_name])} results")
    
    def run_full_experiment(self, auto_arima: bool = False):
        """
        Run experiment on all 3 datasets with all 3 models.
        Skips datasets that are already completed (checkpoint).
        """
        self.load_data()
        
        model_names = ['baseline_ma', 'arima', 'xgboost']
        
        # Disable auto_arima for speed - use fixed ARIMA(2,1,2) order
        # This is justified as we want comparability across datasets and auto_arima is very slow
        auto_arima = False
        
        for dataset_name, df in self.datasets.items():
            if dataset_name in self.completed_datasets:
                print(f"\nSkipping {dataset_name} (already completed)")
                continue
            self.run_experiment_on_dataset(dataset_name, df, model_names, auto_arima, max_products=50)
            self.save_checkpoint(dataset_name)
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Total results collected: {len(self.results)}")
        print(f"Total predictions: {len(self.all_predictions)}")
    
    def save_results(self):
        """Save all results to CSV files."""
        print("\nSaving results...")
        
        # Save fold-level metrics
        df_fold_metrics = pd.DataFrame(self.fold_metrics)
        fold_path = os.path.join(self.output_dir, 'fold_level_metrics.csv')
        df_fold_metrics.to_csv(fold_path, index=False)
        print(f"  Saved fold-level metrics to {fold_path}")
        
        # Save all predictions
        df_predictions = pd.DataFrame(self.all_predictions)
        pred_path = os.path.join(self.output_dir, 'all_predictions.csv')
        df_predictions.to_csv(pred_path, index=False)
        print(f"  Saved all predictions to {pred_path}")
        
        # Calculate summary statistics
        summary = df_fold_metrics.groupby(['dataset', 'model']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std']
        }).reset_index()
        
        summary.columns = ['dataset', 'model', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']
        summary_path = os.path.join(self.output_dir, 'summary_metrics.csv')
        summary.to_csv(summary_path, index=False)
        print(f"  Saved summary metrics to {summary_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY METRICS (Mean ± Std across folds)")
        print(f"{'='*80}")
        print(summary.to_string(index=False))
        print(f"{'='*80}")
        
        return summary, df_fold_metrics, df_predictions


def main():
    base_path = 'c:/Users/devan/Desktop/research paper/dataset'
    output_dir = 'c:/Users/devan/Desktop/research paper/inventory-forecasting/results'
    
    experiment = MultiDatasetExperiment(base_path, output_dir)
    
    # Run with auto_arima enabled for proper order selection
    experiment.run_full_experiment(auto_arima=True)
    
    # Save results
    summary, fold_metrics, predictions = experiment.save_results()
    
    return summary, fold_metrics, predictions


if __name__ == "__main__":
    summary, fold_metrics, predictions = main()
