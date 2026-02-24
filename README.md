<!-- # 📊 Customer Segmentation & Retention Analysis

## 🎯 Project Overview
This project focuses on building an end-to-end Data Science pipeline to identify high-value customers and predict potential churn. By combining behavioral grouping (Clustering) with predictive modeling (Classification), this system provides actionable insights for targeted marketing and customer retention strategies.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Algorithms:** K-Means Clustering, Random Forest Classifier

## 🚀 Project Pipeline

The project is structured into 4 main phases:

### 1. Data Collection & Preprocessing
* Ingested raw transactional data (Online Retail II dataset).
* Cleaned missing values, handled duplicate records, and filtered out anomalies (e.g., negative quantities).
* Engineered a `TotalAmount` feature and treated outliers using the IQR method.

### 2. RFM Modeling
* Transformed raw transaction logs into three behavioral metrics per customer:
  * **Recency:** Days since last purchase.
  * **Frequency:** Total number of unique transactions.
  * **Monetary:** Total revenue generated.
* Applied Quintile Scoring (1-5) to rank customers based on their RFM values.

### 3. K-Means Clustering (Segmentation) - *In Progress*
* Utilizing the Elbow Method and Silhouette Score to determine the optimal number of distinct customer personas (e.g., Champions, Loyal Customers, At-Risk).

### 4. Predictive Modeling (Churn Prediction) - *Upcoming*
* Building a Random Forest classification model to predict which customers are highly likely to churn.
* Extracting Feature Importance to understand the primary drivers of customer attrition.

## 📂 Project Structure

```text
customer-segmentation/
├── data/                  # Contains raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── src/                   # Production-ready Python scripts
│   ├── preprocessing.py   # Data cleaning pipeline
│   ├── rfm.py             # RFM calculation logic
│   ├── clustering.py      # K-Means implementation
│   └── model.py           # Predictive modeling
├── reports/               # Final business summaries and presentations
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation -->


# 📊 Customer Segmentation & Retention Analysis (End-to-End ML Pipeline)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Web App](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

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