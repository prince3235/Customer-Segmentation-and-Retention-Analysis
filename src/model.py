"""
model.py
========
Churn Prediction: XGBoost + SMOTE
──────────────────────────────────
Uses the RFM-enriched DataFrame (with churn label from rfm.py) to:
  1. Build a feature matrix (RFM scores + behavioural features)
  2. Balance the target with SMOTE (handles class imbalance)
  3. Train an XGBoostClassifier with cross-validation
  4. Evaluate with a full classification report & ROC curve
  5. Save the model artifact and feature-importance chart
  6. Expose a predict_churn() function for the Streamlit dashboard
"""

import os
import logging
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve,
)
from sklearn.pipeline         import Pipeline

from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"
MODEL_PATH  = os.path.join(REPORTS_DIR, "churn_xgb_model.pkl")
FI_PATH     = os.path.join(REPORTS_DIR, "feature_importance.png")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. FEATURE MATRIX
# ─────────────────────────────────────────────

MODEL_FEATURES = [
    "recency", "frequency", "monetary",
    "r_score", "f_score", "m_score", "rfm_score",
    "log_recency", "log_frequency", "log_monetary",
]

def build_feature_matrix(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target y from the RFM table.
    Only rows where churn label exists are used.
    """
    available = [f for f in MODEL_FEATURES if f in rfm.columns]
    missing   = [f for f in MODEL_FEATURES if f not in rfm.columns]
    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")

    X = rfm[available].fillna(0)
    y = rfm["churn"]

    logger.info(
        f"Feature matrix: {X.shape}  |  "
        f"Churn rate: {y.mean() * 100:.1f}%  |  "
        f"Features: {available}"
    )
    return X, y


# ─────────────────────────────────────────────
# 2. SMOTE OVERSAMPLING
# ─────────────────────────────────────────────

def apply_smote(X_train: np.ndarray,
                y_train: pd.Series,
                random_state: int = 42) -> tuple:
    """
    Apply SMOTE to the training split to balance the minority class.
    Returns balanced (X_res, y_res).
    """
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    before = dict(pd.Series(y_train).value_counts())
    after  = dict(pd.Series(y_res).value_counts())
    logger.info(f"SMOTE: before={before}  →  after={after}")
    return X_res, y_res


# ─────────────────────────────────────────────
# 3. TRAIN XGBOOST
# ─────────────────────────────────────────────

def train_xgboost(X_res: np.ndarray,
                  y_res: np.ndarray,
                  feature_names: list) -> XGBClassifier:
    """
    Train XGBoostClassifier with tuned hyperparameters.
    5-fold cross-validation AUC is logged.
    """
    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.5,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
    )

    # 5-fold CV AUC on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring="roc_auc")
    logger.info(
        f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  |  "
        f"Per fold: {np.round(cv_scores, 4)}"
    )

    model.fit(X_res, y_res)
    logger.info("XGBoost training complete")
    return model


# ─────────────────────────────────────────────
# 4. EVALUATE MODEL
# ─────────────────────────────────────────────

def evaluate_model(model: XGBClassifier,
                   X_test: np.ndarray,
                   y_test: pd.Series,
                   feature_names: list,
                   save_plots: bool = True) -> dict:
    """
    Full evaluation:
      - Classification report
      - ROC-AUC & PR-AUC
      - Confusion matrix plot
      - ROC curve plot
    Returns a dict with all metrics.
    """
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]

    roc_auc   = roc_auc_score(y_test, y_proba)
    pr_auc    = average_precision_score(y_test, y_proba)
    report    = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}")

    if save_plots:
        _plot_confusion(y_test, y_pred)
        _plot_roc(y_test, y_proba, roc_auc)
        plot_feature_importance(model, feature_names)

    return {
        "roc_auc":  round(roc_auc, 4),
        "pr_auc":   round(pr_auc, 4),
        "report":   report,
    }


def _plot_confusion(y_test, y_pred):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Confusion matrix saved")


def _plot_roc(y_test, y_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")
    ax.plot(fpr, tpr, color="#E63946", linewidth=2.5,
            label=f"XGBoost (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curve – Churn Prediction", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1A1D2E")
    ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("ROC curve saved")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(model: XGBClassifier,
                             feature_names: list,
                             top_n: int = 10,
                             save_path: str = FI_PATH):
    """
    Horizontal bar chart of XGBoost feature importances.
    """
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    colors = ["#E63946" if i == len(fi_df) - 1 else "#457B9D"
              for i in range(len(fi_df))]
    ax.barh(fi_df["feature"], fi_df["importance"], color=colors, alpha=0.9)

    ax.set_xlabel("Importance Score", color="white")
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(axis="x", alpha=0.15, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Feature importance chart saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────
# 6. SAVE / LOAD MODEL
# ─────────────────────────────────────────────

def save_model(model: XGBClassifier,
               scaler: StandardScaler,
               feature_names: list,
               path: str = MODEL_PATH):
    """Pickle the model bundle {model, scaler, feature_names}."""
    bundle = {"model": model, "scaler": scaler, "feature_names": feature_names}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Model bundle saved → {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    """Load a saved model bundle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return bundle


# ─────────────────────────────────────────────
# 7. PREDICT (for dashboard / new data)
# ─────────────────────────────────────────────

def predict_churn(rfm_new: pd.DataFrame,
                  bundle: dict = None,
                  model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Score new customers for churn probability.

    Parameters
    ----------
    rfm_new    : DataFrame with the same features as MODEL_FEATURES
    bundle     : Pre-loaded model bundle dict (optional, loads from disk if None)

    Returns
    -------
    DataFrame with columns: customer_id, churn_prob, churn_prediction
    """
    if bundle is None:
        bundle = load_model(model_path)

    model          = bundle["model"]
    feature_names  = bundle["feature_names"]

    X = rfm_new[[f for f in feature_names if f in rfm_new.columns]].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    result = rfm_new[["customer_id"]].copy()
    result["churn_prob"]       = np.round(proba, 4)
    result["churn_prediction"] = pred
    result["risk_label"] = pd.cut(
        result["churn_prob"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"],
    )
    return result


# ─────────────────────────────────────────────
# 8. MASTER PIPELINE
# ─────────────────────────────────────────────

def run_model_pipeline(rfm: pd.DataFrame,
                       test_size: float = 0.2) -> dict:
    """
    Full ML pipeline:
      1. Build features
      2. Train/test split
      3. SMOTE
      4. Train XGBoost
      5. Evaluate
      6. Save model
    """
    logger.info("=" * 55)
    logger.info("  STARTING MODEL PIPELINE")
    logger.info("=" * 55)

    X, y               = build_feature_matrix(rfm)
    feature_names      = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    logger.info(
        f"Split: train={len(X_train)}  test={len(X_test)}  "
        f"churn_train={y_train.mean():.2%}  churn_test={y_test.mean():.2%}"
    )

    # Scale
    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

    # SMOTE
    Xtr_res, ytr_res = apply_smote(Xtr_sc, y_train)

    # Train
    model  = train_xgboost(Xtr_res, ytr_res, feature_names)

    # Evaluate
    metrics = evaluate_model(model, Xte_sc, y_test, feature_names)

    # Save
    save_model(model, scaler, feature_names)

    logger.info("=" * 55)
    logger.info(
        f"  MODEL PIPELINE COMPLETE  |  "
        f"ROC-AUC={metrics['roc_auc']}  PR-AUC={metrics['pr_auc']}"
    )
    logger.info("=" * 55)
    return {"model": model, "scaler": scaler,
            "feature_names": feature_names, "metrics": metrics}


# ─────────────────────────────────────────────
# 9. STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import run_preprocessing
    from src.rfm           import build_rfm_table

    path    = sys.argv[1] if len(sys.argv) > 1 else "data/online_retail_II.csv"
    df      = run_preprocessing(path)
    rfm     = build_rfm_table(df)
    results = run_model_pipeline(rfm)

    print(f"\n✅ ROC-AUC : {results['metrics']['roc_auc']}")
    print(f"✅ PR-AUC  : {results['metrics']['pr_auc']}")
    print(f"✅ Model saved → {MODEL_PATH}")