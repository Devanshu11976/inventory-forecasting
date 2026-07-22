import sqlite3
import numpy as np
import pandas as pd

# Constants for product archetypes (Section III-B)
GROCERY_STAPLE = {
    "product_id": "prod_grocery_01",
    "category": "grocery",
    "unit_cost": 2.50,
    "base_demand": 100.0,
    "trend": 0.05,
    "weekly_amplitude": 10.0,
    "yearly_amplitude": 5.0,
    "noise_std": 10.0,
    "stockout_rate": 0.015,
}

SEASONAL_APPAREL = {
    "product_id": "prod_apparel_01",
    "category": "apparel",
    "unit_cost": 45.00,
    "base_demand": 30.0,
    "trend": 0.01,
    "weekly_amplitude": 2.0,
    "yearly_amplitude": 15.0,
    "noise_std": 8.0,
    "stockout_rate": 0.03,
}

HARDWARE_ITEM = {
    "product_id": "prod_hardware_01",
    "category": "hardware",
    "unit_cost": 12.00,
    "base_demand": 15.0,
    "trend": 0.002,
    "weekly_amplitude": 0.5,
    "yearly_amplitude": 1.0,
    "noise_std": 2.0,
    "stockout_rate": 0.02,
}

ARCHETYPES = [GROCERY_STAPLE, SEASONAL_APPAREL, HARDWARE_ITEM]

def generate_demand(
    n_days=730,
    base_demand=100.0,
    trend=0.05,
    weekly_amplitude=10.0,
    yearly_amplitude=5.0,
    noise_std=10.0,
    stockout_rate=0.015,
    seed=42
):
    """
    Generates synthetic demand time series using Equation 1:
    D_t = B + beta * t + A_w * sin(2pi * t / 7) + A_y * sin(2pi * t / 365) + e_t
    with demand clipped at 0 and random stockouts.
    
    Args:
        n_days (int): Length of the generated time series.
        base_demand (float): Base demand level B.
        trend (float): Linear trend coefficient beta.
        weekly_amplitude (float): Weekly seasonality amplitude A_w.
        yearly_amplitude (float): Yearly seasonality amplitude A_y.
        noise_std (float): Standard deviation of normal noise sigma.
        stockout_rate (float): Percentage of days forced to 0 (simulated stockouts).
        seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Evaluated demand series.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    
    # Seasonality terms
    weekly_season = weekly_amplitude * np.sin(2.0 * np.pi * t / 7.0)
    yearly_season = yearly_amplitude * np.sin(2.0 * np.pi * t / 365.0)
    
    # Error term
    epsilon = rng.normal(0.0, noise_std, size=n_days)
    
    # Demand calculation
    demand = base_demand + trend * t + weekly_season + yearly_season + epsilon
    
    # Clip demand at 0 (Section III-B: Clip demand at zero)
    demand = np.clip(demand, 0.0, None)
    
    # Simulate stockouts: force days to 0 based on stockout_rate (1.5-3% category-dependent)
    stockout_mask = rng.random(size=n_days) < stockout_rate
    demand[stockout_mask] = 0.0
    
    # Round to integer values for inventory sales units
    return np.round(demand).astype(int)

def populate_db_with_synthetic_data(db_path, n_days=730, seed=42):
    """
    Generates synthetic sales history for all archetypes and inserts into SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert products
    for arch in ARCHETYPES:
        cursor.execute(
            """
            INSERT OR REPLACE INTO products (product_id, category, unit_cost)
            VALUES (?, ?, ?)
            """,
            (arch["product_id"], arch["category"], arch["unit_cost"])
        )
        
    # Generate dates (starting from 2024-01-01)
    dates = pd.date_range(start="2024-01-01", periods=n_days).strftime("%Y-%m-%d")
    
    for i, arch in enumerate(ARCHETYPES):
        # We vary the seed per archetype to prevent identical noise patterns, but keep it deterministic
        arch_seed = seed + i
        sales = generate_demand(
            n_days=n_days,
            base_demand=arch["base_demand"],
            trend=arch["trend"],
            weekly_amplitude=arch["weekly_amplitude"],
            yearly_amplitude=arch["yearly_amplitude"],
            noise_std=arch["noise_std"],
            stockout_rate=arch["stockout_rate"],
            seed=arch_seed
        )
        
        # Prepare list of tuples for batch insert
        sales_records = [
            (arch["product_id"], dates[t], int(sales[t]))
            for t in range(n_days)
        ]
        
        cursor.executemany(
            """
            INSERT OR REPLACE INTO sales_history (product_id, date, units_sold)
            VALUES (?, ?, ?)
            """,
            sales_records
        )
        
    conn.commit()
    conn.close()
