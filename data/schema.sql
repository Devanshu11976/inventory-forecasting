-- Schema for AI-Based Inventory Forecasting System SQLite Database

-- Products table containing metadata about inventory items
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    unit_cost REAL NOT NULL,
    reorder_point REAL
);

-- Sales history table containing daily units sold
CREATE TABLE IF NOT EXISTS sales_history (
    product_id TEXT NOT NULL,
    date TEXT NOT NULL,
    units_sold INTEGER NOT NULL,
    PRIMARY KEY (product_id, date),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Forecasts table containing predictions and corresponding actuals for models
CREATE TABLE IF NOT EXISTS forecasts (
    product_id TEXT NOT NULL,
    date TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_units REAL NOT NULL,
    actual_units REAL,
    PRIMARY KEY (product_id, date, model_name),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
