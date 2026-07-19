import argparse
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from data.generator import populate_db_with_synthetic_data
from datasets.real_data_loader import load_rossmann_to_sqlite
from evaluation.walk_forward import run_walk_forward_evaluation
from evaluation.metrics import mae, rmse, mape, smape
from evaluation.significance import paired_t_test, diebold_mariano_test
from decision.reorder_point import calculate_reorder_point, run_worked_example

def init_database(db_path, schema_path):
    """
    Initializes the SQLite database with the tables defined in schema.sql.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())

def get_all_products(db_path):
    """
    Retrieves all product records from the products table.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM products", conn)
    return df

def get_sales_history_for_product(db_path, product_id):
    """
    Retrieves sorted daily sales records for a single product.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM sales_history WHERE product_id = ?",
            conn,
            params=(product_id,)
        )
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def save_forecasts_to_db(db_path, results_df):
    """
    Saves predictions and actual sales into the SQLite forecasts table.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        records = [
            (row['product_id'], row['date'], row['model_name'], float(row['predicted_units']), float(row['actual_units']))
            for _, row in results_df.iterrows()
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO forecasts (product_id, date, model_name, predicted_units, actual_units)
            VALUES (?, ?, ?, ?, ?)
            """,
            records
        )
        conn.commit()

def save_forecast_plot(actual, predicted, dates, output_path):
    """
    Generates Fig. 2 (Actual vs. XGBoost Predicted) with high aesthetic quality.
    """
    plt.figure(figsize=(12, 6))
    
    # Elegant, clear styling with premium color choices
    plt.plot(dates, actual, label='Actual Sales', color='#2b5c8f', linewidth=2, marker='o', markersize=4)
    plt.plot(dates, predicted, label='XGBoost Forecast', color='#e85d04', linewidth=2, linestyle='--', marker='x', markersize=4)
    
    plt.title('Grocery Staple Sales: Actual vs. XGBoost Predicted (60-Day Horizon)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Units Sold', fontsize=12, labelpad=10)
    
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nFig 2 plot saved successfully to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="AI-Based Inventory Forecasting Experiment Orchestrator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--real-data", action="store_true", help="Run experiment on real Rossmann dataset")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning for ARIMA and XGBoost")
    parser.add_argument("--store-ids", type=int, nargs="+", default=[1, 2, 3], help="Rossmann store IDs to run (default: [1, 2, 3])")
    
    args = parser.parse_args()
    
    # Paths setup
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'inventory.db')
    SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'data', 'schema.sql')
    
    # 1. Init DB
    print("Initializing SQLite database...")
    init_database(DB_PATH, SCHEMA_PATH)
    
    # 2. Ingest Data
    if args.real_data:
        train_csv = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'train.csv'))
        store_csv = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'store.csv'))
        print(f"Loading real Rossmann dataset from {train_csv}...")
        load_rossmann_to_sqlite(train_csv, store_csv, DB_PATH, store_ids=args.store_ids, limit_days=730)
    else:
        print(f"Generating synthetic dataset using seed {args.seed}...")
        populate_db_with_synthetic_data(DB_PATH, n_days=730, seed=args.seed)
        
    # 3. Retrieve products
    products_df = get_all_products(DB_PATH)
    print(f"Loaded products: {list(products_df['product_id'].values)}")
    
    model_names = ['baseline_ma', 'arima', 'xgboost']
    results_list = []
    
    # Default lead times matching archetype characteristics (Section VII)
    lead_time_defaults = {
        'prod_grocery_01': 3,
        'prod_apparel_01': 7,
        'prod_hardware_01': 10
    }
    
    for _, prod in products_df.iterrows():
        product_id = prod['product_id']
        category = prod['category']
        unit_cost = prod['unit_cost']
        
        lead_time = lead_time_defaults.get(product_id, 5)
        
        print(f"\nProcessing product {product_id} ({category}), unit cost ${unit_cost:.2f}, lead time {lead_time} days...")
        df_sales = get_sales_history_for_product(DB_PATH, product_id)
        
        product_forecasts = {}
        for m_name in model_names:
            print(f"  Running model: {m_name}...")
            res_df = run_walk_forward_evaluation(df_sales, m_name, tune=args.tune)
            save_forecasts_to_db(DB_PATH, res_df)
            
            act = res_df['actual_units'].values
            pred = res_df['predicted_units'].values
            
            m_mae = mae(act, pred)
            m_rmse = rmse(act, pred)
            m_mape = mape(act, pred)
            m_smape = smape(act, pred)
            
            # Forecast demand during lead time and safety stock
            forecasted_demand, safety_stock, reorder_point = calculate_reorder_point(pred, m_rmse, lead_time)
            
            product_forecasts[m_name] = {
                'mae': m_mae,
                'rmse': m_rmse,
                'mape': m_mape,
                'smape': m_smape,
                'predictions': pred,
                'actuals': act,
                'forecasted_demand': forecasted_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point
            }
            
            results_list.append({
                'product_id': product_id,
                'category': category,
                'model_name': m_name,
                'mae': m_mae,
                'rmse': m_rmse,
                'mape': m_mape,
                'smape': m_smape,
                'lead_time': lead_time,
                'forecasted_demand': forecasted_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'unit_cost': unit_cost
            })
            
        # Significance Testing
        xgb_data = product_forecasts['xgboost']
        base_data = product_forecasts['baseline_ma']
        
        t_stat, t_pval = paired_t_test(xgb_data['actuals'], xgb_data['predictions'], base_data['predictions'])
        dm_stat, dm_pval = diebold_mariano_test(xgb_data['actuals'], xgb_data['predictions'], base_data['predictions'], h=1)
        
        print(f"  Significance Tests (XGBoost vs Baseline MA):")
        print(f"    Paired t-test: t-stat = {t_stat:.4f}, p-value = {t_pval:.4f}")
        print(f"    Diebold-Mariano test (h=1): DM-stat = {dm_stat:.4f}, p-value = {dm_pval:.4f}")
        
        for r in results_list:
            if r['product_id'] == product_id and r['model_name'] == 'xgboost':
                r['t_stat'] = t_stat
                r['t_pval'] = t_pval
                r['dm_stat'] = dm_stat
                r['dm_pval'] = dm_pval
                
        # Generate plot for Grocery Staple (Fig. 2)
        if product_id == 'prod_grocery_01' or (args.real_data and product_id == str(args.store_ids[0])):
            plot_path = os.path.join(SCRIPT_DIR, 'grocery_staple_forecast.png')
            df_sales_sorted = df_sales.sort_values('date').reset_index(drop=True)
            test_dates = df_sales_sorted['date'].iloc[-60:].dt.strftime('%m-%d').values
            save_forecast_plot(xgb_data['actuals'], xgb_data['predictions'], test_dates, plot_path)
            
    df_results = pd.DataFrame(results_list)
    
    # Save best reorder points (XGBoost) back to products table
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for r in results_list:
            if r['model_name'] == 'xgboost':
                cursor.execute(
                    "UPDATE products SET reorder_point = ? WHERE product_id = ?",
                    (r['reorder_point'], r['product_id'])
                )
        conn.commit()
        
    # Table II (Average performance metrics)
    table_ii = df_results.groupby('model_name')[['mae', 'rmse', 'mape', 'smape']].mean().reset_index()
    print("\n" + "="*80)
    print("TABLE II: OVERALL PERFORMANCE METRICS (AVERAGE ACROSS CATEGORIES)")
    print("="*80)
    print(table_ii.to_markdown(index=False))
    print("="*80)
    
    # Table III (Per-category breakdown)
    table_iii = df_results[[
        'product_id', 'category', 'model_name', 'mae', 'rmse', 'mape', 'smape', 'reorder_point', 'safety_stock'
    ]].copy()
    print("\n" + "="*80)
    print("TABLE III: PER-CATEGORY MODEL COMPARISON AND REORDER ANALYSIS")
    print("="*80)
    print(table_iii.to_markdown(index=False))
    print("="*80)
    
    # Save CSV files
    table_ii_path = os.path.join(SCRIPT_DIR, 'data', 'table_ii.csv')
    table_iii_path = os.path.join(SCRIPT_DIR, 'data', 'table_iii.csv')
    table_ii.to_csv(table_ii_path, index=False)
    table_iii.to_csv(table_iii_path, index=False)
    print(f"\nSaved Table II to {table_ii_path}")
    print(f"Saved Table III to {table_iii_path}")
    
    # Section V-A: worked example
    print("\n" + "="*80)
    print("SECTION V-A: SAFETY STOCK REDUCTION WORKED EXAMPLE")
    print("="*80)
    example = run_worked_example()
    print(f"Unit Cost: ${example['unit_cost']:.2f}")
    print(f"Lead Time: {example['lead_time']} days")
    print(f"Service Level: {example['z']} (~95% Cycle Service Level)")
    print(f"Baseline MA RMSE: {example['rmse_baseline']:.2f}")
    print(f"XGBoost RMSE: {example['rmse_xgboost']:.2f}")
    print(f"Required Safety Stock (Baseline MA): {example['ss_baseline_units']:.2f} units")
    print(f"Required Safety Stock (XGBoost): {example['ss_xgboost_units']:.2f} units")
    print(f"Safety Stock Reduction: {example['reduction_units']:.2f} units")
    print(f"Working Capital Released: ${example['savings_dollars']:.2f} (Target: $3,000.00)")
    print("="*80)
    
    # Financial impact assessment on experimental data
    print("\n" + "="*80)
    print("WORKING CAPITAL IMPACT REPORT")
    print("="*80)
    total_savings = 0.0
    for _, prod in products_df.iterrows():
        prod_id = prod['product_id']
        unit_cost = prod['unit_cost']
        
        xgb_row = df_results[(df_results['product_id'] == prod_id) & (df_results['model_name'] == 'xgboost')].iloc[0]
        base_row = df_results[(df_results['product_id'] == prod_id) & (df_results['model_name'] == 'baseline_ma')].iloc[0]
        
        savings = (base_row['safety_stock'] - xgb_row['safety_stock']) * unit_cost
        total_savings += savings
        print(f"Product: {prod_id} | Unit Cost: ${unit_cost:.2f} | SS Red: {base_row['safety_stock'] - xgb_row['safety_stock']:.2f} units | Savings: ${savings:.2f}")
        
    print(f"Total Working Capital Savings: ${total_savings:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
