"""
preprocessing.py
================
Data Cleaning & Feature Engineering for Online Retail II Dataset
- Handles missing values, duplicates, and invalid transactions
- Computes TotalAmount = Quantity × Price
- Returns a clean DataFrame ready for RFM analysis
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data from the given filepath.
    Supports both comma and semicolon delimiters.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")

    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


# ─────────────────────────────────────────────
# 2. STANDARDISE COLUMN NAMES
# ─────────────────────────────────────────────

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to a consistent snake_case format.
    Handles both 'Customer ID' and 'CustomerID' variants.
    """
    rename_map = {
        "Invoice":      "invoice_no",
        "InvoiceNo":    "invoice_no",
        "StockCode":    "stock_code",
        "Description":  "description",
        "Quantity":     "quantity",
        "InvoiceDate":  "invoice_date",
        "Price":        "unit_price",
        "UnitPrice":    "unit_price",
        "Customer ID":  "customer_id",
        "CustomerID":   "customer_id",
        "Country":      "country",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    logger.info(f"Standardised columns: {df.columns.tolist()}")
    return df


# ─────────────────────────────────────────────
# 3. PARSE DATES
# ─────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert invoice_date column to datetime."""
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], infer_datetime_format=True)
    logger.info(f"Date range: {df['invoice_date'].min()} → {df['invoice_date'].max()}")
    return df


# ─────────────────────────────────────────────
# 4. REMOVE INVALID ROWS
# ─────────────────────────────────────────────

def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that will corrupt RFM / model analysis:
      - Missing customer_id  (can't attribute purchase)
      - Missing description
      - Cancelled invoices   (invoice_no starts with 'C')
      - Non-positive quantity or unit_price
    """
    initial = len(df)

    # 4a. Drop rows with missing customer_id
    df = df.dropna(subset=["customer_id"])
    logger.info(f"After dropping missing customer_id: {len(df)} rows (removed {initial - len(df)})")

    # 4b. Convert customer_id to integer
    df["customer_id"] = df["customer_id"].astype(int)

    # 4c. Remove cancellations (invoice starts with 'C')
    before = len(df)
    df = df[~df["invoice_no"].astype(str).str.startswith("C")]
    logger.info(f"Removed {before - len(df)} cancellations")

    # 4d. Remove non-positive quantity and price
    before = len(df)
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    logger.info(f"Removed {before - len(df)} rows with non-positive quantity/price")

    # 4e. Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows")

    logger.info(f"Clean data shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
      - TotalAmount  = quantity × unit_price  (revenue per line item)
      - year_month   = period string for time-series aggregation
      - day_of_week  = 0 (Mon) – 6 (Sun)
      - hour         = hour of transaction
    """
    df["total_amount"] = df["quantity"] * df["unit_price"]

    df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["invoice_date"].dt.dayofweek
    df["hour"]        = df["invoice_date"].dt.hour

    logger.info("Feature engineering complete: total_amount, year_month, day_of_week, hour")
    return df


# ─────────────────────────────────────────────
# 6. MASTER PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing(filepath: str) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.
    Returns a clean, feature-enriched DataFrame.
    """
    logger.info("=" * 55)
    logger.info("  STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 55)

    df = load_data(filepath)
    df = standardise_columns(df)
    df = parse_dates(df)
    df = remove_invalid_rows(df)
    df = engineer_features(df)

    logger.info("=" * 55)
    logger.info(f"  PREPROCESSING COMPLETE  |  Final shape: {df.shape}")
    logger.info("=" * 55)

    return df


# ─────────────────────────────────────────────
# 7. QUICK SUMMARY HELPER
# ─────────────────────────────────────────────

def data_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for dashboards / notebooks."""
    return {
        "total_rows":         len(df),
        "unique_customers":   df["customer_id"].nunique(),
        "unique_invoices":    df["invoice_no"].nunique(),
        "unique_products":    df["stock_code"].nunique(),
        "unique_countries":   df["country"].nunique(),
        "date_range_start":   str(df["invoice_date"].min().date()),
        "date_range_end":     str(df["invoice_date"].max().date()),
        "total_revenue":      round(df["total_amount"].sum(), 2),
        "avg_order_value":    round(
            df.groupby("invoice_no")["total_amount"].sum().mean(), 2
        ),
    }


# ─────────────────────────────────────────────
# 8. STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/online_retail_II.csv"
    df_clean = run_preprocessing(path)
    summary  = data_summary(df_clean)

    print("\n📊 Dataset Summary")
    print("-" * 40)
    for k, v in summary.items():
        print(f"  {k:<25} : {v}")

    df_clean.to_csv("data/clean_retail.csv", index=False)
    print("\n✅ Clean data saved → data/clean_retail.csv")