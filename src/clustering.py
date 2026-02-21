import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def load_rfm_data(filepath):
    print(f"Loading RFM data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_for_kmeans(df):
    rfm_features = df[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_log = np.log1p(rfm_features) # Log Transform
    
    scaler = StandardScaler()
    scaler.fit(rfm_log)
    rfm_scaled = scaler.transform(rfm_log)
    
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=df.index, columns=rfm_features.columns)
    return rfm_scaled_df, scaler

def find_optimal_k_elbow(scaled_df, max_k=10, save_path="../reports/elbow_plot.png"):
    sse = {}
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_df)
        sse[k] = kmeans.inertia_
        
    plt.figure(figsize=(10, 6))
    plt.plot(list(sse.keys()), list(sse.values()), marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal $k$')
    plt.xlabel('Number of Clusters ($k$)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def apply_kmeans(df, scaled_df, k=4):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_df)
    df['Cluster'] = kmeans.labels_
    return df

def name_clusters(df):
    """
    Mapping numerical clusters to Business Personas based on our data output.
    """
    persona_mapping = {
        0: 'Lost / Hibernating',
        1: 'Champions',
        2: 'Loyal / Promising',
        3: 'At-Risk'
    }
    df['Customer_Persona'] = df['Cluster'].map(persona_mapping)
    print("\nAssigned Business Personas to clusters.")
    return df

def summarize_clusters(df):
    cluster_summary = df.groupby(['Cluster', 'Customer_Persona']).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2)
    
    print("\n--- Final Cluster Profile Summary ---")
    print(cluster_summary)
    return cluster_summary

if __name__ == "__main__":
    INPUT_PATH = "../data/processed/rfm_summary.csv"
    OUTPUT_PATH = "../data/processed/rfm_clusters.csv"
    
    try:
        df = load_rfm_data(INPUT_PATH)
        scaled_data, scaler_obj = preprocess_for_kmeans(df)
        
        # Applying K-Means
        final_df = apply_kmeans(df, scaled_data, k=4)
        
        # Adding Business Names
        final_df = name_clusters(final_df)
        
        # Summary
        summarize_clusters(final_df)
        
        # Save
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSUCCESS: Phase 3 Completed! Data saved to {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\nError: {e}")