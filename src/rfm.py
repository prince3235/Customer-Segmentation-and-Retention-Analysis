import pandas as pd
import numpy as np
import os

def load_clean_data(filepath):
    """
    Loads the clean transaction data from Phase 1.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading clean data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert Date column back to datetime (CSV loses this format)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def calculate_rfm(df):
    """
    Computes Recency, Frequency, and Monetary values for each customer.
    """
    # 1. Determine Snapshot Date (The day after the last purchase in the dataset)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Snapshot Date: {snapshot_date}")

    # 2. Group by CustomerID and calculate metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                  # Frequency (Count unique orders)
        'TotalAmount': 'sum'                                     # Monetary
    }).reset_index()

    # 3. Rename columns
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)

    print(f"RFM Table Shape: {rfm.shape}")
    return rfm

def score_rfm(rfm):
    """
    Assigns a score from 1-5 for each metric.
    Recency: Lower is better (5 = most recent).
    Frequency/Monetary: Higher is better (5 = highest spend/freq).
    """
    # Create labels for scores (1 to 5)
    labels = range(1, 6) # [1, 2, 3, 4, 5]
    
    # Recency Score (Reverse order: Lower days = Higher Score)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=sorted(labels, reverse=True))
    
    # Frequency Score (Higher count = Higher Score)
    # Note: We use 'rank' method='first' because many customers might have Frequency=1, causing duplicate edges
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=labels)
    
    # Monetary Score (Higher money = Higher Score)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=labels)
    
    # Combine into a single string (e.g., "555" is a champion)
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Calculate Data Type Sum for clustering later
    rfm['RFM_Score_Sum'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    print("RFM Scoring Complete.")
    return rfm

if __name__ == "__main__":
    # Define Paths
    INPUT_PATH = "../data/processed/clean_transactions.csv"
    OUTPUT_PATH = "../data/processed/rfm_summary.csv"

    try:
        # 1. Load Data
        df = load_clean_data(INPUT_PATH)
        
        # 2. Calculate RFM
        rfm_df = calculate_rfm(df)
        
        # 3. Apply Scoring (1-5)
        rfm_scored = score_rfm(rfm_df)
        
        # 4. Save
        rfm_scored.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSUCCESS: RFM Analysis saved to {OUTPUT_PATH}")
        print(rfm_scored.head())

    except Exception as e:
        print(f"\nError: {e}")