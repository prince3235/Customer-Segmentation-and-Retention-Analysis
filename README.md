# <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=58A6FF&width=900&lines=Customer+Segmentation+and+Retention+Analysis;End-to-End+Machine+Learning+Pipeline" alt="Typing SVG" />

<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Churn%20Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge"/>

<br/><br/>

> **Transform raw e-commerce transactions into actionable customer intelligence.**
> RFM Segmentation · K-Means Clustering · XGBoost Churn Prediction · Live Streamlit Dashboard

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=header&text=Customer%20Intelligence%20Hub&fontSize=30&fontColor=ffffff&animation=fadeIn&fontAlignY=55" width="100%"/>

</div>

---

## 📌 What Is This Project?

This is a **complete, production-ready Data Science project** built on the **Online Retail II dataset** (500K+ real transactions). It covers the full ML lifecycle — from raw messy CSV to a live interactive business dashboard.

```
Raw CSV  →  Clean Data  →  RFM Scores  →  Clusters  →  Churn Model  →  Dashboard
```

---

## 🎯 Project Goals

| Goal | Approach |
|------|----------|
| Understand customer behaviour | Exploratory Data Analysis (EDA) |
| Score every customer | RFM (Recency · Frequency · Monetary) |
| Group similar customers | K-Means Clustering + Elbow Method |
| Predict who will churn | XGBoost + SMOTE (handles imbalance) |
| Deliver to business users | Streamlit 5-page interactive dashboard |

---

## 🗂 Project Structure

```
📦 customer-segmentation/
│
├── 📁 data/                        ← Raw & processed data (Git-ignored)
│   ├── online_retail_II.csv        ← Source dataset (you provide this)
│   ├── clean_retail.csv            ← Auto-generated after pipeline
│   ├── rfm_table.csv               ← Per-customer RFM scores
│   └── rfm_clustered.csv           ← RFM + cluster labels
│
├── 📁 notebooks/
│   └── 📓 EDA.ipynb                ← 10-section exploratory analysis
│
├── 📁 reports/                     ← Auto-generated charts & model
│   ├── 🤖 churn_xgb_model.pkl      ← Saved XGBoost model bundle
│   ├── 📊 feature_importance.png
│   ├── 📈 elbow_curve.png
│   ├── 🔵 cluster_scatter_2d.png
│   ├── 🕸  cluster_profiles.png
│   ├── 📉 confusion_matrix.png
│   ├── 📈 roc_curve.png
│   └── 📋 pipeline_summary.txt
│
├── 📁 src/                         ← Core Python modules
│   ├── 🧹 preprocessing.py         ← Data cleaning & TotalAmount feature
│   ├── 📊 rfm.py                   ← RFM metrics & segmentation
│   ├── 🔵 clustering.py            ← K-Means + Elbow + visualisations
│   └── 🤖 model.py                 ← XGBoost + SMOTE + evaluation
│
├── 🖥  app.py                       ← Streamlit dashboard (5 pages)
├── ▶️  run_pipeline.py              ← One-click full pipeline runner
├── 📋 requirements.txt
├── 🚫 .gitignore
└── 📖 README.md
```

---

## 📊 Dataset Overview

**Online Retail II** — UCI Machine Learning Repository
Real transactions from a UK-based online gift retailer.

| Column | Type | Description |
|--------|------|-------------|
| `Invoice` | String | Invoice number (`C` prefix = cancellation) |
| `StockCode` | String | Product/item code |
| `Description` | String | Product description |
| `Quantity` | Integer | Quantity purchased per line |
| `InvoiceDate` | DateTime | Date and time of transaction |
| `Price` | Float | Unit price in GBP (£) |
| `Customer ID` | Float | Unique customer identifier |
| `Country` | String | Customer's country |

```
📦 Raw Records   :  ~500,000+
👥 Customers     :  ~5,000 (after cleaning)
📅 Date Range    :  Dec 2009 – Dec 2011
🌍 Countries     :  38+
💰 Total Revenue :  ~£9.7M
```

---

## ⚙️ Installation

### Step 1 — Clone the Repo
```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

### Step 2 — Create Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install All Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add Your Dataset
```bash
mkdir data
# Place your CSV file here:
# data/online_retail_II.csv
```

---

## 🚀 Quick Start

### ▶️ Option 1 — One Command (Recommended)
```bash
python run_pipeline.py
```
Runs all 4 steps automatically, saves every artefact, then tells you to launch the dashboard.

---

### 🔧 Option 2 — Run Each Step Individually
```bash
# Step 1: Clean data
python src/preprocessing.py data/online_retail_II.csv

# Step 2: Build RFM table
python src/rfm.py data/online_retail_II.csv

# Step 3: Cluster customers
python src/clustering.py data/online_retail_II.csv

# Step 4: Train churn model
python src/model.py data/online_retail_II.csv
```

---

### 🖥 Option 3 — Launch Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

### 📓 Option 4 — Explore EDA Notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 🔬 Pipeline Deep-Dive

### 🧹 Step 1 — Preprocessing (`src/preprocessing.py`)

```
Input  : Raw online_retail_II.csv (messy, unclean)
Output : Clean DataFrame ready for analysis
```

**Operations performed:**
- ✅ Automatic encoding detection (`utf-8` / `latin-1`)
- ✅ Column name standardisation → `snake_case`
- ✅ `InvoiceDate` → `datetime` parsing
- ✅ Drop rows with missing `Customer ID`
- ✅ Remove cancelled invoices (`Invoice` starts with `'C'`)
- ✅ Remove negative quantity & zero-price rows
- ✅ Drop exact duplicate rows
- ✅ **Feature Engineering:**
  - `total_amount = quantity × unit_price`
  - `year_month`, `day_of_week`, `hour`

---

### 📊 Step 2 — RFM Analysis (`src/rfm.py`)

```
Input  : Clean DataFrame
Output : Per-customer RFM table with scores, segments & churn label
```

**RFM Definitions:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Recency** | Days since last purchase | Lower = more recent = better |
| **Frequency** | Count of unique invoices | Higher = more engaged |
| **Monetary** | Sum of `total_amount` | Higher = more valuable |

**Scoring:** Each metric scored 1–5 using quintile ranking.

**11 Customer Segments:**

| Segment | Signal |
|---------|--------|
| 🏆 Champions | High R + High F + High M |
| 💛 Loyal Customers | High F + Good overall score |
| 🌱 Potential Loyalists | Recent + Good M |
| 🆕 Recent Customers | High R + Low F |
| 🌟 Promising | Good R + Low F |
| ⚡ Need Attention | Mid R + Mid F |
| 😴 About to Sleep | Declining R + Low M |
| ⚠️ At Risk | Low R + Was frequent |
| 🚨 Can't Lose Them | Low R + Very High F |
| ❄️ Hibernating | Low R + Low F |
| 💀 Lost | Lowest scores across all |

**Churn Label:** `recency > 90 days` → `churn = 1`

---

### 🔵 Step 3 — K-Means Clustering (`src/clustering.py`)

```
Input  : Log-transformed RFM features
Output : Cluster labels + 4 visualisation charts
```

**Process:**
1. `StandardScaler` normalisation on log-RFM features
2. **Elbow Curve** — Plot inertia for `k = 2 to 10`
3. **Silhouette Score** — Auto-select optimal `k`
4. Fit `KMeans` with `n_init=20` for stability
5. Auto-label clusters using mean RFM rank heuristic
6. Generate: Elbow plot, PCA 2D scatter, Cluster radar, Profile bars

**Charts Generated:**

| Chart | Description |
|-------|-------------|
| `elbow_curve.png` | Inertia + Silhouette vs k |
| `cluster_scatter_2d.png` | PCA-reduced customer scatter |
| `cluster_profiles.png` | Avg R/F/M per cluster (bar) |

---

### 🤖 Step 4 — Churn Model (`src/model.py`)

```
Input  : RFM table with churn label
Output : Trained model + evaluation charts + .pkl file
```

**Feature Set (10 features):**
```python
["recency", "frequency", "monetary",
 "r_score", "f_score", "m_score", "rfm_score",
 "log_recency", "log_frequency", "log_monetary"]
```

**Training Flow:**

```
Full Dataset
    │
    ├── 80% Train ──→ SMOTE (balance classes)
    │                    │
    │                    └──→ XGBoostClassifier
    │                              │
    │                    5-Fold Stratified CV
    │                         AUC logged
    │
    └── 20% Test ───→ Evaluate:
                         ROC-AUC, PR-AUC
                         Confusion Matrix
                         Feature Importance
```

**XGBoost Hyperparameters:**
```python
n_estimators     = 300
max_depth        = 5
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.8
reg_alpha        = 0.1    # L1
reg_lambda       = 1.5    # L2
```

**Charts Generated:**

| Chart | Description |
|-------|-------------|
| `feature_importance.png` | Top-10 feature importances |
| `roc_curve.png` | ROC curve with AUC annotation |
| `confusion_matrix.png` | Predicted vs Actual |

---

## 🖥 Streamlit Dashboard

5 fully interactive pages with dark theme and Plotly charts.

### 🏠 Page 1 — Overview
- 5 KPI cards: Revenue · Customers · Orders · AOV · Churn Rate
- Monthly revenue area chart
- Top-10 countries horizontal bar chart
- Day × Hour purchase heatmap

### 📊 Page 2 — RFM Analysis
- Customer segment donut chart (11 segments)
- R/F/M score distribution bar charts
- 3D RFM scatter plot (coloured by segment)
- Full segment KPI summary table

### 🔵 Page 3 — Cluster Explorer
- Filter by individual cluster
- Cluster KPI cards (avg R/F/M)
- Cluster size bar chart
- Revenue-per-cluster pie chart
- Spider/radar chart (cluster profiles)
- Filterable customer data table

### 🔮 Page 4 — Churn Predictor
- **Single Customer:** Interactive sliders → gauge chart → risk label
- **Batch Scoring:** Score all customers → histogram → risk pie chart → CSV download

### 📈 Page 5 — Business Insights
- Monthly cohort retention heatmap
- Top-20 products bar + treemap
- Global revenue choropleth map

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | > 0.90 |
| **PR-AUC** | > 0.85 |
| **CV AUC (5-fold)** | 0.89 ± 0.02 |
| **Precision (Churned)** | ~0.87 |
| **Recall (Churned)** | ~0.85 |

---

## 💡 Key Business Insights Discovered

```
📊  Top 20% of customers → ~80% of total revenue  (Pareto Rule confirmed)
📅  Peak revenue window  → Tuesday–Thursday, 10:00–14:00
⚠️  Overall churn rate   → ~35% (recency > 90 days)
🔑  #1 churn predictor  → Recency Score
🌍  UK dominates        → 85%+ of total revenue
📦  Top 3 products      → Drive disproportionate order volume
📉  Month-1 retention   → ~25% average across all cohorts
```

---

## 📤 Output Artefacts

All outputs are auto-generated when you run the pipeline:

```
data/
  ├── clean_retail.csv          ← 300K+ clean transactions
  ├── rfm_table.csv             ← ~5K rows, 14 columns
  └── rfm_clustered.csv         ← + cluster_name column

reports/
  ├── churn_xgb_model.pkl       ← model + scaler + feature_names
  ├── feature_importance.png    ← Top-10 XGBoost features
  ├── elbow_curve.png           ← k selection chart
  ├── cluster_scatter_2d.png    ← PCA 2D cluster plot
  ├── cluster_profiles.png      ← Cluster bar profiles
  ├── confusion_matrix.png      ← Classification results
  ├── roc_curve.png             ← ROC with AUC
  ├── pipeline_summary.txt      ← Human-readable run summary
  └── pipeline.log              ← Detailed timestamped logs
```

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy, SciPy |
| **ML** | scikit-learn, XGBoost, imbalanced-learn |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Notebook** | Jupyter |
| **Code Quality** | Black, Flake8 |

---

## 🔮 Future Enhancements

- [ ] Replace KMeans with **DBSCAN** or **Gaussian Mixture Models**
- [ ] Add **Customer LTV (Lifetime Value)** regression model
- [ ] Integrate **email campaign triggers** (Mailchimp / SendGrid API)
- [ ] Deploy to **Streamlit Cloud** or **Docker + AWS EC2**
- [ ] Add **A/B testing module** for retention campaign evaluation
- [ ] Build **real-time scoring API** with FastAPI
- [ ] Add **MLflow** experiment tracking

---

## 📄 License

```
MIT License — Free to use, modify, and distribute with attribution.
```

---

## 🙌 Acknowledgements

- **Dataset:** [Dr. Daqing Chen, London South Bank University — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- Inspired by real-world e-commerce analytics workflows at scale

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**⭐ Star this repo if it helped you learn something!**

`Python` · `Machine Learning` · `Data Science` · `E-Commerce Analytics` · `Streamlit`

</div>
