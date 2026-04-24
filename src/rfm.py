"""
rfm.py
======
RFM (Recency · Frequency · Monetary) Metric Calculation
─────────────────────────────────────────────────────────
Given the clean retail DataFrame produced by preprocessing.py,
this module computes per-customer RFM scores, assigns segments,
and returns a tidy DataFrame ready for clustering and modelling.

RFM Definitions
───────────────
  Recency   – Days since the customer's LAST purchase
               (lower = more recent = better)
  Frequency – Total number of UNIQUE invoices
               (higher = more engaged)
  Monetary  – Total revenue contributed  (Σ total_amount)
               (higher = more valuable)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. COMPUTE RAW RFM VALUES
# ─────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame,
                snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute raw Recency, Frequency, Monetary values per customer.

    Parameters
    ----------
    df            : Clean retail DataFrame (output of preprocessing.run_preprocessing)
    snapshot_date : Reference date for Recency.
                    Defaults to max(invoice_date) + 1 day.

    Returns
    -------
    rfm : DataFrame indexed by customer_id with columns
          [recency, frequency, monetary]
    """
    if snapshot_date is None:
        snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)

    logger.info(f"Snapshot date for Recency: {snapshot_date.date()}")

    rfm = (
        df.groupby("customer_id")
        .agg(
            last_purchase  = ("invoice_date",  "max"),
            frequency      = ("invoice_no",    "nunique"),
            monetary       = ("total_amount",  "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days
    rfm = rfm.drop(columns=["last_purchase"])

    logger.info(
        f"RFM computed for {len(rfm):,} customers | "
        f"Recency  μ={rfm['recency'].mean():.1f}d | "
        f"Frequency μ={rfm['frequency'].mean():.1f} | "
        f"Monetary  μ=£{rfm['monetary'].mean():.2f}"
    )
    return rfm


# ─────────────────────────────────────────────
# 2. SCORE RFM (1-5 quintile ranking)
# ─────────────────────────────────────────────

def score_rfm(rfm: pd.DataFrame,
              q: int = 5) -> pd.DataFrame:
    """
    Assign quintile-based scores (1 = worst, 5 = best) to each dimension.

    Recency   → lower recency days = better → reversed labels
    Frequency → higher = better
    Monetary  → higher = better
    """
    rfm = rfm.copy()

    # Recency: lower days → higher score (reversed)
    rfm["r_score"] = pd.qcut(
        rfm["recency"], q=q,
        labels=list(range(q, 0, -1)),
        duplicates="drop"
    ).astype(int)

    # Frequency
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), q=q,
        labels=list(range(1, q + 1)),
        duplicates="drop"
    ).astype(int)

    # Monetary
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), q=q,
        labels=list(range(1, q + 1)),
        duplicates="drop"
    ).astype(int)

    # Composite RFM Score (simple average)
    rfm["rfm_score"] = (rfm["r_score"] + rfm["f_score"] + rfm["m_score"]) / 3

    logger.info("RFM scores assigned (1–5 quintiles)")
    return rfm


# ─────────────────────────────────────────────
# 3. SEGMENT CUSTOMERS
# ─────────────────────────────────────────────

SEGMENT_MAP = {
    # Champions
    "Champions":          lambda r: (r["r_score"] >= 4) & (r["f_score"] >= 4),
    # Loyal Customers
    "Loyal Customers":    lambda r: (r["f_score"] >= 3) & (r["rfm_score"] >= 3),
    # Potential Loyalists
    "Potential Loyalists":lambda r: (r["r_score"] >= 3) & (r["f_score"] <= 3) & (r["m_score"] >= 3),
    # Recent Customers
    "Recent Customers":   lambda r: (r["r_score"] >= 4) & (r["f_score"] <= 2),
    # Promising
    "Promising":          lambda r: (r["r_score"] >= 3) & (r["f_score"] <= 2),
    # Need Attention
    "Need Attention":     lambda r: (r["r_score"] <= 3) & (r["f_score"] <= 3) & (r["rfm_score"] >= 2.5),
    # About to Sleep
    "About to Sleep":     lambda r: (r["r_score"] <= 3) & (r["f_score"] <= 2) & (r["m_score"] <= 3),
    # At Risk
    "At Risk":            lambda r: (r["r_score"] <= 2) & (r["f_score"] >= 3),
    # Can't Lose Them
    "Can't Lose Them":    lambda r: (r["r_score"] <= 2) & (r["f_score"] >= 4),
    # Hibernating
    "Hibernating":        lambda r: (r["r_score"] <= 2) & (r["f_score"] <= 2),
    # Lost
    "Lost":               lambda r: r["rfm_score"] <= 1.5,
}

def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a human-readable segment label to each customer
    based on RFM scores using a priority-ordered rule set.
    """
    rfm = rfm.copy()
    rfm["segment"] = "Others"

    for segment, condition in SEGMENT_MAP.items():
        mask = condition(rfm) & (rfm["segment"] == "Others")
        rfm.loc[mask, "segment"] = segment

    counts = rfm["segment"].value_counts()
    logger.info(f"Segment distribution:\n{counts.to_string()}")
    return rfm


# ─────────────────────────────────────────────
# 4. CHURN LABEL
# ─────────────────────────────────────────────

def add_churn_label(rfm: pd.DataFrame,
                    recency_threshold: int = 90) -> pd.DataFrame:
    """
    Binary churn label:
      1 → Churned   (recency > threshold days)
      0 → Active    (recency ≤ threshold days)

    Default threshold = 90 days (3 months of inactivity)
    """
    rfm = rfm.copy()
    rfm["churn"] = (rfm["recency"] > recency_threshold).astype(int)

    churn_rate = rfm["churn"].mean() * 100
    logger.info(
        f"Churn label added (threshold={recency_threshold}d) | "
        f"Churn rate: {churn_rate:.1f}%"
    )
    return rfm


# ─────────────────────────────────────────────
# 5. LOG-TRANSFORM FOR SKEWED FEATURES
# ─────────────────────────────────────────────

def log_transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transform to reduce right-skew in monetary
    and frequency features (used before clustering / training).
    """
    rfm = rfm.copy()
    rfm["log_recency"]   = np.log1p(rfm["recency"])
    rfm["log_frequency"] = np.log1p(rfm["frequency"])
    rfm["log_monetary"]  = np.log1p(rfm["monetary"])
    logger.info("Log1p transforms added: log_recency, log_frequency, log_monetary")
    return rfm


# ─────────────────────────────────────────────
# 6. MASTER PIPELINE
# ─────────────────────────────────────────────

def build_rfm_table(df: pd.DataFrame,
                    snapshot_date: pd.Timestamp = None,
                    churn_threshold: int = 90) -> pd.DataFrame:
    """
    End-to-end RFM pipeline:
      raw RFM → scores → segments → churn label → log features
    """
    logger.info("=" * 55)
    logger.info("  STARTING RFM PIPELINE")
    logger.info("=" * 55)

    rfm = compute_rfm(df, snapshot_date)
    rfm = score_rfm(rfm)
    rfm = assign_segments(rfm)
    rfm = add_churn_label(rfm, churn_threshold)
    rfm = log_transform_rfm(rfm)

    logger.info("=" * 55)
    logger.info(f"  RFM COMPLETE  |  Shape: {rfm.shape}")
    logger.info("=" * 55)
    return rfm


# ─────────────────────────────────────────────
# 7. SEGMENT SUMMARY HELPER
# ─────────────────────────────────────────────

def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot table: segment → count, avg_recency,
    avg_frequency, avg_monetary, churn_rate.
    """
    summary = (
        rfm.groupby("segment")
        .agg(
            customer_count = ("customer_id",  "count"),
            avg_recency    = ("recency",       "mean"),
            avg_frequency  = ("frequency",     "mean"),
            avg_monetary   = ("monetary",      "mean"),
            churn_rate     = ("churn",         "mean"),
        )
        .round(2)
        .sort_values("avg_monetary", ascending=False)
        .reset_index()
    )
    summary["churn_rate"] = (summary["churn_rate"] * 100).round(1)
    return summary


# ─────────────────────────────────────────────
# 8. STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import run_preprocessing

    path    = sys.argv[1] if len(sys.argv) > 1 else "data/online_retail_II.csv"
    df      = run_preprocessing(path)
    rfm     = build_rfm_table(df)

    print("\n📊 RFM Table (first 5 rows)")
    print(rfm.head().to_string())

    print("\n📊 Segment Summary")
    print(segment_summary(rfm).to_string())

    rfm.to_csv("data/rfm_table.csv", index=False)
    print("\n✅ RFM table saved → data/rfm_table.csv")