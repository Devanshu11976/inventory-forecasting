import argparse
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Clears any existing database file to prevent stale/mixed data.
    """
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
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

def get_all_sales_history(db_path):
    """
    Retrieves all sales records from sales_history table.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM sales_history", conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
    return df

def process_single_product(product_id, category, unit_cost, df_sales, model_names, tune, lead_time_defaults):
    lead_time = lead_time_defaults.get(product_id, 5)
    
    product_results = []
    product_forecasts_to_save = []
    
    product_forecasts = {}
    for m_name in model_names:
        res_df = run_walk_forward_evaluation(df_sales, m_name, tune=tune)
        
        act = res_df['actual_units'].values
        pred = res_df['predicted_units'].values
        
        m_mae = mae(act, pred)
        m_rmse = rmse(act, pred)
        m_mape = mape(act, pred)
        m_smape = smape(act, pred)
        
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
        
        product_results.append({
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
        
        for _, row in res_df.iterrows():
            product_forecasts_to_save.append((
                row['product_id'],
                row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                row['model_name'],
                float(row['predicted_units']),
                float(row['actual_units'])
            ))
            
    xgb_data = product_forecasts['xgboost']
    base_data = product_forecasts['baseline_ma']
    
    try:
        t_stat, t_pval = paired_t_test(xgb_data['actuals'], xgb_data['predictions'], base_data['predictions'])
    except Exception:
        t_stat, t_pval = float('nan'), float('nan')
        
    try:
        dm_stat, dm_pval = diebold_mariano_test(xgb_data['actuals'], xgb_data['predictions'], base_data['predictions'], h=1)
    except Exception:
        dm_stat, dm_pval = float('nan'), float('nan')
        
    for r in product_results:
        if r['model_name'] == 'xgboost':
            r['t_stat'] = t_stat
            r['t_pval'] = t_pval
            r['dm_stat'] = dm_stat
            r['dm_pval'] = dm_pval
            
    return product_results, product_forecasts_to_save

def main():
    parser = argparse.ArgumentParser(description="AI-Based Inventory Forecasting Experiment Orchestrator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--synthetic-data", action="store_true", help="Run experiment on synthetic dataset instead of real Rossmann dataset")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning for ARIMA and XGBoost")
    parser.add_argument("--store-ids", type=int, nargs="+", default=[1, 2, 3], help="Rossmann store IDs to run (default: [1, 2, 3])")
    parser.add_argument("--all-stores", action="store_true", help="Run experiment on all Rossmann stores")
    
    args = parser.parse_args()
    
    # Paths setup
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'inventory.db')
    SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'data', 'schema.sql')
    
    # 1. Init DB
    print("Initializing SQLite database...")
    init_database(DB_PATH, SCHEMA_PATH)
    
    # 2. Ingest Data
    if not args.synthetic_data:
        train_csv = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'train.csv'))
        store_csv = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'store.csv'))
        store_ids = 'all' if args.all_stores else args.store_ids
        print(f"Loading real Rossmann dataset from {train_csv}...")
        load_rossmann_to_sqlite(train_csv, store_csv, DB_PATH, store_ids=store_ids, limit_days=730)
    else:
        print(f"Generating synthetic dataset using seed {args.seed}...")
        populate_db_with_synthetic_data(DB_PATH, n_days=730, seed=args.seed)
        
    # 3. Retrieve products
    products_df = get_all_products(DB_PATH)
    if len(products_df) > 10:
        print(f"Loaded {len(products_df)} products.")
    else:
        print(f"Loaded products: {list(products_df['product_id'].values)}")
    
    model_names = ['baseline_ma', 'arima', 'xgboost']
    results_list = []
    forecasts_records_to_save = []
    
    # Default lead times matching archetype characteristics (Section VII)
    lead_time_defaults = {
        'prod_grocery_01': 3,
        'prod_apparel_01': 7,
        'prod_hardware_01': 10
    }
    
    print("Retrieving sales history for processing...")
    df_all_sales = get_all_sales_history(DB_PATH)
    sales_by_product = {pid: group for pid, group in df_all_sales.groupby('product_id')}
    
    valid_products_df = products_df[products_df['product_id'].isin(sales_by_product.keys())].copy()
    
    # Run walk-forward evaluations in parallel
    print(f"Running evaluation on {len(valid_products_df)} products in parallel using ProcessPoolExecutor...")
    futures = {}
    with ProcessPoolExecutor() as executor:
        for _, prod in valid_products_df.iterrows():
            product_id = prod['product_id']
            category = prod['category']
            unit_cost = prod['unit_cost']
            df_sales = sales_by_product[product_id]
            
            future = executor.submit(
                process_single_product,
                product_id,
                category,
                unit_cost,
                df_sales,
                model_names,
                args.tune,
                lead_time_defaults
            )
            futures[future] = product_id
            
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            product_id = futures[future]
            completed += 1
            try:
                prod_results, prod_forecasts = future.result()
                results_list.extend(prod_results)
                forecasts_records_to_save.extend(prod_forecasts)
                if completed % 50 == 0 or completed == total or (total <= 10):
                    print(f"  Processed {completed}/{total} products...")
            except Exception as e:
                print(f"  Error processing product {product_id}: {e}")
                
    # Save forecasts
    if forecasts_records_to_save:
        print("Saving forecasts to database...")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO forecasts (product_id, date, model_name, predicted_units, actual_units)
                VALUES (?, ?, ?, ?, ?)
                """,
                forecasts_records_to_save
            )
            conn.commit()
            
    # Save best reorder points
    if results_list:
        print("Saving reorder points to database...")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            for r in results_list:
                if r['model_name'] == 'xgboost':
                    cursor.execute(
                        "UPDATE products SET reorder_point = ? WHERE product_id = ?",
                        (r['reorder_point'], r['product_id'])
                    )
            conn.commit()

    # Generate plot for Grocery Staple (Fig. 2)
    target_plot_product = 'prod_grocery_01' if args.synthetic_data else str(args.store_ids[0]) if not args.all_stores else '1'
    
    target_results = [r for r in results_list if r['product_id'] == target_plot_product]
    if target_results and forecasts_records_to_save:
        df_sales = sales_by_product[target_plot_product]
        df_sales_sorted = df_sales.sort_values('date').reset_index(drop=True)
        test_dates = df_sales_sorted['date'].iloc[-60:].dt.strftime('%m-%d').values
        
        target_forecasts = [f for f in forecasts_records_to_save if f[0] == target_plot_product and f[2] == 'xgboost']
        target_forecasts = sorted(target_forecasts, key=lambda x: x[1])
        actuals = [f[4] for f in target_forecasts]
        predictions = [f[3] for f in target_forecasts]
        
        plot_path = os.path.join(SCRIPT_DIR, 'grocery_staple_forecast.png')
        save_forecast_plot(actuals, predictions, test_dates, plot_path)
        
    df_results = pd.DataFrame(results_list)
    
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
    if len(table_iii) > 30:
        print(table_iii.head(30).to_markdown(index=False))
        print(f"\n... and {len(table_iii) - 30} more rows (full table saved to data/table_iii.csv)")
    else:
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
    printed_count = 0
    for _, prod in products_df.iterrows():
        prod_id = prod['product_id']
        unit_cost = prod['unit_cost']
        
        xgb_rows = df_results[(df_results['product_id'] == prod_id) & (df_results['model_name'] == 'xgboost')]
        base_rows = df_results[(df_results['product_id'] == prod_id) & (df_results['model_name'] == 'baseline_ma')]
        
        if len(xgb_rows) > 0 and len(base_rows) > 0:
            xgb_row = xgb_rows.iloc[0]
            base_row = base_rows.iloc[0]
            
            savings = (base_row['safety_stock'] - xgb_row['safety_stock']) * unit_cost
            total_savings += savings
            
            if len(products_df) <= 10 or printed_count < 10 or prod_id in ['1', '2', '3', 'prod_grocery_01', 'prod_apparel_01', 'prod_hardware_01']:
                print(f"Product: {prod_id} | Unit Cost: ${unit_cost:.2f} | SS Red: {base_row['safety_stock'] - xgb_row['safety_stock']:.2f} units | Savings: ${savings:.2f}")
                printed_count += 1
                
    if len(products_df) > 10:
        print(f"... and {len(products_df) - printed_count} more products.")
    print(f"Total Working Capital Savings: ${total_savings:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
