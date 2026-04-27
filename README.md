# <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=26&pause=1000&color=58A6FF&width=900&lines=Customer+Segmentation+%26+Retention+Analysis;End-to-End+Machine+Learning+Pipeline" alt="Typing SVG" />

<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Churn%20Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge"/>

<br/><br/>

> **500K+ raw transactions → RFM Segments → K-Means Clusters → XGBoost Churn Model → Live Dashboard**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=header&text=Customer%20Intelligence%20Hub&fontSize=28&fontColor=ffffff&animation=fadeIn&fontAlignY=55" width="100%"/>

</div>

---

## 📌 Overview

A **production-ready, end-to-end ML pipeline** on the **Online Retail II dataset** that transforms raw e-commerce data into actionable customer intelligence.

```
Raw CSV → Clean Data → RFM Scores → K-Means Clusters → Churn Model → Streamlit Dashboard
```

| What | How |
|------|-----|
| Customer Scoring | RFM Analysis — Recency · Frequency · Monetary |
| Segmentation | K-Means + Elbow + Silhouette (auto k) |
| Churn Prediction | XGBoost + SMOTE · ROC-AUC > 0.90 |
| Business Dashboard | Streamlit — 5 interactive pages · Dark theme |

---

## 🗂 Project Structure

```
📦 customer-segmentation/
├── 📁 data/               ← Git-ignored (add CSV here)
├── 📁 notebooks/
│   └── EDA.ipynb          ← 10-section exploratory analysis
├── 📁 reports/            ← Auto-generated charts + model .pkl
├── 📁 src/
│   ├── __init__.py
│   ├── preprocessing.py   ← Cleaning + TotalAmount feature
│   ├── rfm.py             ← RFM scores + 11 segments + churn label
│   ├── clustering.py      ← K-Means + Elbow + PCA plots
│   └── model.py           ← XGBoost + SMOTE + evaluation
├── app.py                 ← Streamlit dashboard (5 pages)
├── run_pipeline.py        ← One-click full pipeline
├── config.py              ← All settings in one place
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone & enter
git clone https://github.com/prince3235/Customer-Segmentation-and-Retention-Analysis.git
cd Customer-Segmentation-and-Retention-Analysis

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset → data/online_retail_II.csv

# 5. Run full pipeline
python run_pipeline.py

# 6. Launch dashboard
streamlit run app.py
```

---

## 🔬 Pipeline at a Glance

| Step | File | What It Does |
|------|------|--------------|
| 1️⃣ Preprocessing | `src/preprocessing.py` | Clean data, remove cancellations, engineer `TotalAmount` |
| 2️⃣ RFM Analysis | `src/rfm.py` | Score customers 1–5, assign 11 segments, label churn |
| 3️⃣ Clustering | `src/clustering.py` | Auto-select k, fit KMeans, generate PCA + radar charts |
| 4️⃣ Churn Model | `src/model.py` | SMOTE balance, train XGBoost, 5-fold CV, save `.pkl` |

---

## 🖥 Dashboard — 5 Pages

| Page | Highlights |
|------|-----------|
| 🏠 Overview | Revenue trend · Country chart · Time heatmap · 5 KPIs |
| 📊 RFM Analysis | Segment donut · 3D scatter · Score distributions |
| 🔵 Cluster Explorer | Radar chart · Filter by cluster · Customer table |
| 🔮 Churn Predictor | Live gauge · Batch scoring · CSV download |
| 📈 Business Insights | Cohort retention · Product treemap · World map |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | **> 0.90** |
| PR-AUC | **> 0.85** |
| CV AUC (5-fold) | **0.89 ± 0.02** |

---

## 💡 Key Business Findings

```
📊 Top 20% customers   →  ~80% of total revenue   (Pareto confirmed)
⚠️  Churn rate          →  ~35%  (recency > 90 days)
🔑 #1 churn predictor  →  Recency Score
📅 Peak revenue time   →  Tue–Thu · 10:00–14:00
🌍 UK dominance        →  85%+ of total revenue
```

---

## 🛠 Tech Stack

`Python 3.10` · `Pandas` · `XGBoost` · `scikit-learn` · `imbalanced-learn` · `Streamlit` · `Plotly` · `Matplotlib` · `Seaborn` · `Jupyter`

---

## 🔮 Future Work

- [ ] DBSCAN / Gaussian Mixture Models for clustering
- [ ] Customer LTV regression model
- [ ] FastAPI real-time scoring endpoint
- [ ] Deploy → Streamlit Cloud / Docker + AWS
- [ ] MLflow experiment tracking

---

## 🙌 Acknowledgements

Dataset by [Dr. Daqing Chen — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**⭐ Star this repo if it helped you!**

`Machine Learning` · `Data Science` · `E-Commerce Analytics` · `Streamlit`

</div>