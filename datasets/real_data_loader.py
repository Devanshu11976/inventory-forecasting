import sqlite3
import pandas as pd
import numpy as np

def load_rossmann_to_sqlite(train_csv_path, store_csv_path, db_path, store_ids=[1, 2, 3], limit_days=730):
    """
    Ingests Rossmann Store Sales CSV into the SQLite database matching the sales_history schema.
    No data transformation is performed during raw ingestion (Section II / Section VIII).
    
    Args:
        train_csv_path (str): Path to Rossmann train.csv.
        store_csv_path (str): Path to Rossmann store.csv.
        db_path (str): Path to output SQLite database.
        store_ids (list): List of store IDs to ingest.
        limit_days (int): Maximum number of days of history to load per store.
    """
    # 1. Read files
    df_train = pd.read_csv(train_csv_path)
    df_store = pd.read_csv(store_csv_path)
    
    # 2. Filter by selected store IDs
    df_train = df_train[df_train['Store'].isin(store_ids)].copy()
    df_store = df_store[df_store['Store'].isin(store_ids)].copy()
    
    # 3. Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Map StoreType to unit cost
    type_cost_map = {'a': 10.0, 'b': 15.0, 'c': 20.0, 'd': 25.0}
    
    # 4. Ingest products
    for _, row in df_store.iterrows():
        store_id = str(int(row['Store']))
        store_type = row['StoreType']
        category = f"store_type_{store_type}"
        unit_cost = type_cost_map.get(store_type, 15.0)
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO products (product_id, category, unit_cost)
            VALUES (?, ?, ?)
            """,
            (store_id, category, unit_cost)
        )
        
    # 5. Ingest sales history
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train = df_train.sort_values(['Store', 'Date'])
    
    total_loaded = 0
    for store_id in store_ids:
        store_sales = df_train[df_train['Store'] == store_id]
        
        # Take the tail limit_days of sales history
        store_sales = store_sales.tail(limit_days)
        
        sales_records = []
        for _, row in store_sales.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            units_sold = int(row['Sales'])
            sales_records.append((str(store_id), date_str, units_sold))
            
        cursor.executemany(
            """
            INSERT OR REPLACE INTO sales_history (product_id, date, units_sold)
            VALUES (?, ?, ?)
            """,
            sales_records
        )
        total_loaded += len(sales_records)
        
    conn.commit()
    conn.close()
    print(f"Successfully loaded {len(store_ids)} stores ({total_loaded} sales rows) from Rossmann dataset into SQLite.")
