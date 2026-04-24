"""
run_pipeline.py
===============
One-Click Full Pipeline Runner
────────────────────────────────
Executes the complete Customer Segmentation & Retention Analysis pipeline:
  Step 1 → Data Preprocessing & Feature Engineering
  Step 2 → RFM Metric Calculation & Segmentation
  Step 3 → K-Means Clustering (auto k via Elbow + Silhouette)
  Step 4 → XGBoost Churn Model (SMOTE + CV)
  Step 5 → Save all artefacts (model, charts, CSVs)

Usage:
  python run_pipeline.py                              (uses default path)
  python run_pipeline.py data/online_retail_II.csv    (custom path)
"""

import sys
import os
import time
import logging

# ── Logging setup ─────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reports/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs("reports", exist_ok=True)
os.makedirs("data",    exist_ok=True)

sys.path.insert(0, ".")


# ──────────────────────────────────────────────
# STEP BANNER HELPER
# ──────────────────────────────────────────────

def banner(step: int, title: str):
    width = 60
    logger.info("=" * width)
    logger.info(f"  STEP {step}  |  {title}")
    logger.info("=" * width)


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"


# ──────────────────────────────────────────────
# STEP 1 – PREPROCESSING
# ──────────────────────────────────────────────

def step1_preprocessing(data_path: str):
    banner(1, "DATA PREPROCESSING & FEATURE ENGINEERING")
    t0 = time.time()

    from src.preprocessing import run_preprocessing, data_summary
    df = run_preprocessing(data_path)

    # Save clean CSV
    clean_path = "data/clean_retail.csv"
    df.to_csv(clean_path, index=False)

    summary = data_summary(df)
    logger.info(f"  ✅ Clean data saved → {clean_path}")
    logger.info(f"  📦 Shape         : {df.shape}")
    logger.info(f"  👥 Customers     : {summary['unique_customers']:,}")
    logger.info(f"  🛒 Invoices      : {summary['unique_invoices']:,}")
    logger.info(f"  💰 Total Revenue : £{summary['total_revenue']:,.2f}")
    logger.info(f"  ⏱  Elapsed       : {elapsed(t0)}")

    return df


# ──────────────────────────────────────────────
# STEP 2 – RFM
# ──────────────────────────────────────────────

def step2_rfm(df):
    banner(2, "RFM METRIC CALCULATION & SEGMENTATION")
    t0 = time.time()

    from src.rfm import build_rfm_table, segment_summary
    rfm = build_rfm_table(df)

    # Save
    rfm_path = "data/rfm_table.csv"
    rfm.to_csv(rfm_path, index=False)

    summary = segment_summary(rfm)
    logger.info(f"  ✅ RFM table saved → {rfm_path}")
    logger.info(f"  📊 Segments:\n{summary[['segment','customer_count','churn_rate']].to_string(index=False)}")
    logger.info(f"  ⏱  Elapsed : {elapsed(t0)}")

    return rfm


# ──────────────────────────────────────────────
# STEP 3 – CLUSTERING
# ──────────────────────────────────────────────

def step3_clustering(rfm):
    banner(3, "K-MEANS CLUSTERING  (Elbow + Silhouette)")
    t0 = time.time()

    from src.clustering import run_clustering
    rfm_clustered, km, scaler = run_clustering(rfm)

    # Save
    cluster_path = "data/rfm_clustered.csv"
    rfm_clustered.to_csv(cluster_path, index=False)

    logger.info(f"  ✅ Clustered RFM saved → {cluster_path}")
    cluster_summary = rfm_clustered["cluster_name"].value_counts()
    for name, cnt in cluster_summary.items():
        logger.info(f"     {name:<30} : {cnt:,} customers")
    logger.info(f"  ⏱  Elapsed : {elapsed(t0)}")

    return rfm_clustered


# ──────────────────────────────────────────────
# STEP 4 – CHURN MODEL
# ──────────────────────────────────────────────

def step4_model(rfm_clustered):
    banner(4, "XGBOOST CHURN MODEL  (SMOTE + Cross-Validation)")
    t0 = time.time()

    from src.model import run_model_pipeline
    results = run_model_pipeline(rfm_clustered)
    metrics = results["metrics"]

    logger.info(f"  ✅ Model saved → reports/churn_xgb_model.pkl")
    logger.info(f"  📈 ROC-AUC : {metrics['roc_auc']}")
    logger.info(f"  📈 PR-AUC  : {metrics['pr_auc']}")
    logger.info(f"  ⏱  Elapsed : {elapsed(t0)}")

    return results


# ──────────────────────────────────────────────
# STEP 5 – SUMMARY REPORT
# ──────────────────────────────────────────────

def step5_report(df, rfm_clustered, model_results):
    banner(5, "PIPELINE SUMMARY REPORT")

    lines = [
        "=" * 60,
        "  CUSTOMER SEGMENTATION & RETENTION ANALYSIS",
        "  Pipeline Run Summary",
        "=" * 60,
        "",
        f"  Raw Records        : {len(df):,}",
        f"  Unique Customers   : {df['customer_id'].nunique():,}",
        f"  Total Revenue      : £{df['total_amount'].sum():,.2f}",
        f"  Churn Rate         : {rfm_clustered['churn'].mean()*100:.1f}%",
        f"  K-Means Clusters   : {rfm_clustered['cluster'].nunique()}",
        "",
        "  Model Performance",
        f"    ROC-AUC          : {model_results['metrics']['roc_auc']}",
        f"    PR-AUC           : {model_results['metrics']['pr_auc']}",
        "",
        "  Artefacts Saved",
        "    data/clean_retail.csv",
        "    data/rfm_table.csv",
        "    data/rfm_clustered.csv",
        "    reports/churn_xgb_model.pkl",
        "    reports/feature_importance.png",
        "    reports/elbow_curve.png",
        "    reports/cluster_scatter_2d.png",
        "    reports/cluster_profiles.png",
        "    reports/confusion_matrix.png",
        "    reports/roc_curve.png",
        "",
        "  ✅ Run `streamlit run app.py` to launch dashboard",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    report_path = "reports/pipeline_summary.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info("\n" + report_text)
    logger.info(f"\n📄 Summary saved → {report_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    t_total  = time.time()
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/online_retail_II.csv"

    logger.info("\n" + "█" * 60)
    logger.info("  🚀  CUSTOMER SEGMENTATION PIPELINE  🚀")
    logger.info("█" * 60)
    logger.info(f"  Data path : {data_path}\n")

    if not os.path.exists(data_path):
        logger.error(
            f"❌ File not found: {data_path}\n"
            "   Please provide the Online Retail II CSV file."
        )
        sys.exit(1)

    # ── Run all steps ─────────────────────────
    df              = step1_preprocessing(data_path)
    rfm             = step2_rfm(df)
    rfm_clustered   = step3_clustering(rfm)
    model_results   = step4_model(rfm_clustered)
    step5_report(df, rfm_clustered, model_results)

    logger.info(f"\n🎉 FULL PIPELINE COMPLETE  |  Total time: {elapsed(t_total)}")
    logger.info("   Launch dashboard:  streamlit run app.py\n")


if __name__ == "__main__":
    main()