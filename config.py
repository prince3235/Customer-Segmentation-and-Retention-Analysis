"""
config.py
=========
Central Configuration File
───────────────────────────
All project-wide settings in ONE place.
Change here → affects entire pipeline.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")

# Input
RAW_DATA_PATH   = os.path.join(DATA_DIR, "online_retail_II.csv")

# Processed outputs
CLEAN_DATA_PATH    = os.path.join(DATA_DIR, "clean_retail.csv")
RFM_TABLE_PATH     = os.path.join(DATA_DIR, "rfm_table.csv")
CLUSTERED_RFM_PATH = os.path.join(DATA_DIR, "rfm_clustered.csv")

# Model
MODEL_PATH = os.path.join(REPORTS_DIR, "churn_xgb_model.pkl")

# ─────────────────────────────────────────────
# RFM SETTINGS
# ─────────────────────────────────────────────

RFM_QUANTILES        = 5      # Score range: 1 to 5
CHURN_THRESHOLD_DAYS = 90     # Days inactive → labelled as churned

# ─────────────────────────────────────────────
# CLUSTERING SETTINGS
# ─────────────────────────────────────────────

CLUSTER_K_RANGE   = range(2, 11)   # Try k = 2 to 10
CLUSTER_N_INIT    = 20             # KMeans random restarts
CLUSTER_FEATURES  = ["log_recency", "log_frequency", "log_monetary"]

# ─────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────

TEST_SIZE      = 0.20
RANDOM_STATE   = 42
CV_FOLDS       = 5

XGBOOST_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 5,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 3,
    "gamma"            : 0.1,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.5,
    "use_label_encoder": False,
    "eval_metric"      : "logloss",
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
}

MODEL_FEATURES = [
    "recency", "frequency", "monetary",
    "r_score", "f_score", "m_score", "rfm_score",
    "log_recency", "log_frequency", "log_monetary",
]

# ─────────────────────────────────────────────
# DASHBOARD SETTINGS
# ─────────────────────────────────────────────

PAGE_TITLE  = "Customer Intelligence Hub"
PAGE_ICON   = "🧠"
LAYOUT      = "wide"

# ─────────────────────────────────────────────
# AUTO-CREATE DIRECTORIES
# ─────────────────────────────────────────────

for _dir in [DATA_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(_dir, exist_ok=True)