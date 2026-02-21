import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# Define the standard names we WANT to use
STANDARD_COLUMNS = {
    'Invoice': 'InvoiceNo',
    'InvoiceNo': 'InvoiceNo',
    'StockCode': 'StockCode',
    'Description': 'Description',
    'Quantity': 'Quantity',
    'InvoiceDate': 'InvoiceDate',
    'InvoiceDa': 'InvoiceDate',  # Handle cut-off header
    'Price': 'UnitPrice',
    'UnitPrice': 'UnitPrice',
    'Customer ID': 'CustomerID', # Common variation with space
    'Customer': 'CustomerID',    # From your screenshot
    'CustomerID': 'CustomerID',
    'Country': 'Country'
}

def load_data(filepath):
    """
    Loads raw data and cleans column headers immediately.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    print(f"Loading data from {filepath}...")
    
    # Attempt to read with 'ISO-8859-1' encoding (standard for retail data)
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath)
        
    print(f"Initial shape: {df.shape}")
    
    # --- CRITICAL FIX: Clean Column Names ---
    # 1. Remove leading/trailing spaces (e.g., " Customer ID " -> "Customer ID")
    df.columns = df.columns.str.strip()
    
    print(f"Raw Columns found in CSV: {df.columns.tolist()}")
    return df

def initial_processing(df):
    """
    Renames columns and creates TotalAmount.
    """
    # 1. Rename Columns using the mapping
    # This aligns whatever is in your CSV to our standard names
    df.rename(columns=STANDARD_COLUMNS, inplace=True)
    
    # 2. Check if critical columns exist now
    required = ['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        print(f"Current Columns: {df.columns.tolist()}")
        raise KeyError("Stop: Cannot proceed without CustomerID and Transaction details.")

    # 3. Create 'TotalAmount' (Quantity * UnitPrice)
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    print("Created 'TotalAmount' column.")
    
    return df

def clean_data(df):
    """
    Standard cleaning pipeline.
    """
    # 1. Drop missing CustomerIDs
    initial_count = len(df)
    df = df.dropna(subset=['CustomerID'])
    print(f"Dropped {initial_count - len(df)} rows with missing CustomerID.")

    # 2. Remove Duplicates
    df = df.drop_duplicates()

    # 3. Convert Date
    # Coerce errors to NaT (Not a Time) then drop them
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    
    # 4. Filter Negative/Zero Transactions (Returns/Errors)
    # We only want positive sales for segmentation
    df = df[(df['Quantity'] > 0) & (df['TotalAmount'] > 0)]
    
    print(f"Data shape after cleaning: {df.shape}")
    return df

def handle_outliers(df, col='TotalAmount'):
    """
    Caps outliers using IQR to prevent massive orders from skewing clusters.
    """
    if col in df.columns:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        
        # Cap values between 5th and 95th percentile
        df[col] = df[col].clip(lower=Q1, upper=Q3)
        print(f"Outliers in '{col}' capped between {Q1:.2f} and {Q3:.2f}.")
        
    return df

if __name__ == "__main__":
    # Adjust path to match your folder structure
    RAW_DATA_PATH = "../data/raw/online_retail_II.csv"  
    PROCESSED_DATA_PATH = "../data/processed/clean_transactions.csv"

    try:
        # 1. Load
        df = load_data(RAW_DATA_PATH)
        
        # 2. Rename & Feature Engineer
        df = initial_processing(df)
        
        # 3. Clean
        df = clean_data(df)
        
        # 4. Outliers
        df = handle_outliers(df, col='TotalAmount')
        
        # 5. Save
        # Ensure directory exists
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nSUCCESS: Clean data saved to {PROCESSED_DATA_PATH}")
        
    except Exception as e:
        print(f"\nExecution Failed: {e}")