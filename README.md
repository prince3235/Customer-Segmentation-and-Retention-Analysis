# 📊 Customer Segmentation & Retention Analysis

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
└── README.md              # Project documentation