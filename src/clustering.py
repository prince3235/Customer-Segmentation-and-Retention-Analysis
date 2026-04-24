"""
clustering.py
=============
K-Means Customer Segmentation & Elbow Method
─────────────────────────────────────────────
Uses the log-transformed RFM features from rfm.py to cluster
customers into distinct behaviour groups, then visualises:
  • Elbow curve (inertia vs. k)
  • Silhouette scores
  • 3-D cluster scatter (Recency × Frequency × Monetary)
  • Cluster profile radar chart
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing  import StandardScaler
from sklearn.cluster         import KMeans
from sklearn.metrics         import silhouette_score
from sklearn.decomposition   import PCA

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Colour palette for clusters
CLUSTER_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC", "#6D6875",
]


# ─────────────────────────────────────────────
# 1. PREPARE FEATURES
# ─────────────────────────────────────────────

CLUSTER_FEATURES = ["log_recency", "log_frequency", "log_monetary"]

def prepare_cluster_features(rfm: pd.DataFrame) -> np.ndarray:
    """
    Scale log-transformed RFM features to zero mean / unit variance.
    Returns a NumPy array ready for KMeans.
    """
    X = rfm[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"Feature matrix shape: {X_scaled.shape}")
    return X_scaled, scaler


# ─────────────────────────────────────────────
# 2. ELBOW METHOD
# ─────────────────────────────────────────────

def elbow_method(X_scaled: np.ndarray,
                 k_range: range = range(2, 11),
                 save_path: str = None) -> dict:
    """
    Compute inertia and silhouette score for each k in k_range.

    Returns
    -------
    results : dict with keys 'k_values', 'inertia', 'silhouette'
    """
    inertia    = []
    silhouette = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        silhouette.append(sil)
        logger.info(f"  k={k}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

    results = {
        "k_values":  list(k_range),
        "inertia":   inertia,
        "silhouette": silhouette,
    }

    # ── Plot ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1D2E")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    ax1.plot(results["k_values"], inertia, "o-", color="#E63946", linewidth=2.5, markersize=7)
    ax1.set_title("Elbow Curve (Inertia)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster SSE)")
    ax1.grid(alpha=0.2, color="white")

    ax2.plot(results["k_values"], silhouette, "s-", color="#2A9D8F", linewidth=2.5, markersize=7)
    ax2.set_title("Silhouette Score by k", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(alpha=0.2, color="white")

    fig.suptitle("Optimal k Selection", color="white", fontsize=16, y=1.01)
    plt.tight_layout()

    path = save_path or os.path.join(REPORTS_DIR, "elbow_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Elbow plot saved → {path}")
    results["plot_path"] = path
    return results


# ─────────────────────────────────────────────
# 3. FIT KMEANS
# ─────────────────────────────────────────────

def fit_kmeans(X_scaled: np.ndarray, k: int = 4) -> KMeans:
    """Fit K-Means with the chosen k and return the fitted model."""
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    km.fit(X_scaled)
    logger.info(f"KMeans fitted  k={k}  |  inertia={km.inertia_:.0f}")
    return km


# ─────────────────────────────────────────────
# 4. LABEL CLUSTERS WITH HUMAN NAMES
# ─────────────────────────────────────────────

def label_clusters(rfm: pd.DataFrame,
                   labels: np.ndarray) -> pd.DataFrame:
    """
    Attach numeric cluster labels to the RFM table, then derive
    a human-readable cluster name based on avg RFM characteristics.
    """
    rfm = rfm.copy()
    rfm["cluster"] = labels

    # Compute cluster means on original (non-log) RFM
    profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()

    # Simple heuristic naming
    name_map = {}
    for c, row in profile.iterrows():
        r_rank = (profile["recency"].rank(ascending=True)  [c])   # lower recency = better
        f_rank = (profile["frequency"].rank(ascending=False)[c])
        m_rank = (profile["monetary"].rank(ascending=False) [c])
        avg_rank = (r_rank + f_rank + m_rank) / 3
        n = len(profile)

        if avg_rank <= n * 0.25:
            name_map[c] = "🏆 Champions"
        elif avg_rank <= n * 0.50:
            name_map[c] = "💛 Loyalists"
        elif avg_rank <= n * 0.75:
            name_map[c] = "⚠️  At Risk"
        else:
            name_map[c] = "❄️  Lost / Hibernating"

    rfm["cluster_name"] = rfm["cluster"].map(name_map)
    logger.info(f"Cluster label map: {name_map}")
    logger.info(f"\n{rfm['cluster_name'].value_counts().to_string()}")
    return rfm


# ─────────────────────────────────────────────
# 5. VISUALISE CLUSTERS (PCA 2-D scatter)
# ─────────────────────────────────────────────

def plot_clusters_2d(rfm: pd.DataFrame,
                     X_scaled: np.ndarray,
                     save_path: str = None):
    """PCA-reduced 2-D scatter plot coloured by cluster."""
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var    = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    clusters = sorted(rfm["cluster"].unique())
    for i, c in enumerate(clusters):
        mask = rfm["cluster"] == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=18, alpha=0.65, color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
            label=rfm.loc[rfm["cluster"] == c, "cluster_name"].iloc[0],
        )

    ax.set_title("Customer Clusters (PCA 2-D)", color="white", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)", color="white")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)", color="white")
    ax.tick_params(colors="white")
    legend = ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1A1D2E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    path = save_path or os.path.join(REPORTS_DIR, "cluster_scatter_2d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Cluster scatter saved → {path}")
    return path


# ─────────────────────────────────────────────
# 6. CLUSTER PROFILE BAR CHART
# ─────────────────────────────────────────────

def plot_cluster_profiles(rfm: pd.DataFrame,
                          save_path: str = None):
    """
    Bar chart showing average Recency / Frequency / Monetary per cluster.
    """
    profile = (
        rfm.groupby("cluster_name")[["recency", "frequency", "monetary"]]
        .mean()
        .round(1)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0F1117")

    metrics = [
        ("recency",   "Avg Recency (days)",   "#E63946", "lower = better"),
        ("frequency", "Avg Frequency",         "#2A9D8F", "higher = better"),
        ("monetary",  "Avg Monetary (£)",      "#E9C46A", "higher = better"),
    ]

    for ax, (col, title, color, note) in zip(axes, metrics):
        ax.set_facecolor("#1A1D2E")
        bars = ax.barh(profile.index, profile[col], color=color, alpha=0.85)
        ax.set_title(f"{title}\n({note})", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        for bar, val in zip(bars, profile[col]):
            ax.text(bar.get_width() * 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.1f}", va="center", color="white", fontsize=9)

    fig.suptitle("Cluster Profiles", color="white", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = save_path or os.path.join(REPORTS_DIR, "cluster_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Cluster profiles saved → {path}")
    return path


# ─────────────────────────────────────────────
# 7. MASTER PIPELINE
# ─────────────────────────────────────────────

def run_clustering(rfm: pd.DataFrame,
                   k: int = None,
                   k_range: range = range(2, 11)) -> pd.DataFrame:
    """
    Full clustering pipeline:
      1. Prepare & scale features
      2. Elbow analysis (choose k automatically if not given)
      3. Fit KMeans
      4. Label clusters
      5. Save all plots

    Returns the enriched RFM DataFrame with cluster columns.
    """
    logger.info("=" * 55)
    logger.info("  STARTING CLUSTERING PIPELINE")
    logger.info("=" * 55)

    X_scaled, scaler = prepare_cluster_features(rfm)

    # Elbow analysis
    elbow = elbow_method(X_scaled, k_range)

    # Auto-select k as the one with the highest silhouette score
    if k is None:
        best_idx = int(np.argmax(elbow["silhouette"]))
        k = elbow["k_values"][best_idx]
        logger.info(f"Auto-selected k={k} (highest silhouette={elbow['silhouette'][best_idx]:.4f})")

    # Fit model
    km     = fit_kmeans(X_scaled, k)
    labels = km.predict(X_scaled)

    # Label
    rfm_clustered = label_clusters(rfm, labels)

    # Plots
    plot_clusters_2d(rfm_clustered, X_scaled)
    plot_cluster_profiles(rfm_clustered)

    logger.info("=" * 55)
    logger.info(f"  CLUSTERING COMPLETE  |  k={k}")
    logger.info("=" * 55)

    return rfm_clustered, km, scaler


# ─────────────────────────────────────────────
# 8. STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import run_preprocessing
    from src.rfm           import build_rfm_table

    path     = sys.argv[1] if len(sys.argv) > 1 else "data/online_retail_II.csv"
    df       = run_preprocessing(path)
    rfm      = build_rfm_table(df)
    rfm_seg, km, scaler = run_clustering(rfm)

    print("\n📊 Cluster value counts")
    print(rfm_seg["cluster_name"].value_counts().to_string())

    rfm_seg.to_csv("data/rfm_clustered.csv", index=False)
    print("\n✅ Clustered RFM saved → data/rfm_clustered.csv")