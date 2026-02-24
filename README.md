# 📊 Customer Segmentation & Retention Analysis (End-to-End ML Pipeline)

## 🎯 Executive Summary
In today's competitive e-commerce landscape, retaining existing customers is more cost-effective than acquiring new ones. This project provides an end-to-end Data Science solution that shifts marketing from a "one-size-fits-all" approach to a **hyper-personalized strategy**. 

By engineering an **RFM (Recency, Frequency, Monetary)** framework, clustering customers via **K-Means**, and predicting future churn risk using an **optimized XGBoost classifier**, this system allows businesses to proactively identify high-value customers and rescue at-risk accounts.

---

## 🚀 Key Features & Business Impact
* **Behavioral Segmentation (Unsupervised Learning):** Grouped 1M+ raw transaction records into 4 distinct business personas (Champions, Loyal, At-Risk, Lost) using K-Means clustering.
* **Churn Prediction (Supervised Learning):** Developed an advanced classification model using **XGBoost** and **SMOTE** (for class imbalance), predicting future customer flight risk with **~69% accuracy**.
* **Interactive Web Dashboard:** Deployed a **Streamlit** web application serving as a real-time CRM tool for marketing teams to instantly search customers, view their segment, and check their AI-predicted churn probability.

---

## 🛠️ Tech Stack
* **Data Manipulation & Analysis:** `Pandas`, `NumPy`
* **Machine Learning:** `Scikit-Learn`, `XGBoost`, `Imbalanced-Learn (SMOTE)`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Deployment & UI:** `Streamlit`, `Joblib`

---

## 📂 Project Architecture

```text
customer-segmentation/
├── data/                  # Data directory (Git-ignored for security/size)
├── notebooks/             # Jupyter notebooks for exploratory data analysis
├── reports/               # Generated graphs, charts, and the saved ML model
│   ├── churn_xgb_model.pkl
│   └── feature_importance.png
├── src/                   # Core Python modules
│   ├── preprocessing.py   # Data cleaning & feature engineering (TotalAmount)
│   ├── rfm.py             # RFM metric calculation
│   ├── clustering.py      # K-Means segmentation & Elbow method
│   └── model.py           # XGBoost training & SMOTE implementation
├── app.py                 # Streamlit Web Dashboard application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation