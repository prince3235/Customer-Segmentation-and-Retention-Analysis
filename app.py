"""
app.py
======
Customer Segmentation & Retention Analysis
Streamlit Web Dashboard  –  Full Multi-Page Application
────────────────────────────────────────────────────────
Pages:
  1. 🏠 Overview          – KPIs, revenue trend, country map
  2. 📊 RFM Analysis      – Score distributions, segment donut
  3. 🔵 Cluster Explorer  – Interactive cluster deep-dive
  4. 🔮 Churn Predictor   – Single customer risk scoring
  5. 📈 Business Insights – Cohort analysis, product performance

Run:  streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Streamlit Page Config ───────────────────
st.set_page_config(
    page_title  = "Customer Intelligence Hub",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ──────────────────────────────
st.markdown("""
<style>
/* Dark theme overrides */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
    color: #E6EDF3;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161B22 0%, #0D1117 100%);
    border-right: 1px solid #30363D;
}
.metric-card {
    background: linear-gradient(135deg, #1C2128 0%, #21262D 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #58A6FF; }
.metric-label { font-size: 0.85rem; color: #8B949E; margin-top: 4px; }
.metric-delta { font-size: 0.85rem; margin-top: 4px; }
.section-header {
    font-size: 1.4rem; font-weight: 700;
    color: #E6EDF3; margin: 1.5rem 0 1rem 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    color: #8B949E;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# DATA LOADING  (cached)
# ═══════════════════════════════════════════════════════

@st.cache_data(show_spinner="⚙️ Loading & processing data…")
def load_all_data(data_path: str):
    """Run full pipeline and return (df_clean, rfm, rfm_clustered)."""
    from src.preprocessing import run_preprocessing
    from src.rfm           import build_rfm_table
    from src.clustering    import run_clustering

    df_clean              = run_preprocessing(data_path)
    rfm                   = build_rfm_table(df_clean)
    rfm_clustered, km, sc = run_clustering(rfm)
    return df_clean, rfm_clustered


@st.cache_data(show_spinner="🤖 Training churn model…")
def load_or_train_model(rfm: pd.DataFrame):
    """Load pre-trained model or train fresh."""
    from src.model import load_model, run_model_pipeline, MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        result = run_model_pipeline(rfm)
        return {"model": result["model"], "scaler": result["scaler"],
                "feature_names": result["feature_names"]}


# ═══════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ═══════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(28,33,40,0.8)",
    font          = dict(color="#C9D1D9", family="Inter, sans-serif"),
    xaxis         = dict(gridcolor="#21262D", linecolor="#30363D"),
    yaxis         = dict(gridcolor="#21262D", linecolor="#30363D"),
    margin        = dict(l=20, r=20, t=50, b=20),
)

SEGMENT_COLORS = {
    "Champions":           "#F72585",
    "Loyal Customers":     "#7209B7",
    "Potential Loyalists": "#3A0CA3",
    "Recent Customers":    "#4361EE",
    "Promising":           "#4CC9F0",
    "Need Attention":      "#F9C74F",
    "About to Sleep":      "#F8961E",
    "At Risk":             "#F3722C",
    "Can't Lose Them":     "#E63946",
    "Hibernating":         "#6C757D",
    "Lost":                "#343A40",
    "Others":              "#ADB5BD",
}


# ═══════════════════════════════════════════════════════
# SIDEBAR  – navigation + filters
# ═══════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px 0;">
          <span style="font-size:2.5rem;">🧠</span>
          <h2 style="color:#58A6FF; margin:4px 0; font-size:1.2rem;">
              Customer Intelligence
          </h2>
          <p style="color:#8B949E; font-size:0.75rem; margin:0;">
              End-to-End ML Pipeline
          </p>
        </div>
        <hr style="border-color:#30363D; margin: 12px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["🏠 Overview",
             "📊 RFM Analysis",
             "🔵 Cluster Explorer",
             "🔮 Churn Predictor",
             "📈 Business Insights"],
            label_visibility="collapsed",
        )

        # Data path
        st.markdown("<hr style='border-color:#30363D;'>", unsafe_allow_html=True)
        st.markdown("**⚙️ Configuration**")
        data_path = st.text_input(
            "CSV Path",
            value="data/online_retail_II.csv",
            help="Path to the Online Retail II CSV file",
        )

        st.markdown("<hr style='border-color:#30363D;'>", unsafe_allow_html=True)
        st.caption("Built with ❤️  using Streamlit · XGBoost · K-Means")

    return page, data_path


# ═══════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════

def page_overview(df: pd.DataFrame, rfm: pd.DataFrame):
    st.markdown("## 🏠 Business Overview")
    st.markdown("High-level KPIs and revenue trends across all customers.")

    # ── KPI Row ──────────────────────────────────
    total_rev  = df["total_amount"].sum()
    n_cust     = df["customer_id"].nunique()
    n_orders   = df["invoice_no"].nunique()
    avg_order  = df.groupby("invoice_no")["total_amount"].sum().mean()
    churn_rate = rfm["churn"].mean() * 100

    cols = st.columns(5)
    kpis = [
        ("💰 Total Revenue",    f"£{total_rev:,.0f}",  ""),
        ("👥 Unique Customers", f"{n_cust:,}",          ""),
        ("🛒 Total Orders",     f"{n_orders:,}",        ""),
        ("🧾 Avg Order Value",  f"£{avg_order:,.2f}",  ""),
        ("⚠️ Churn Rate",       f"{churn_rate:.1f}%",  ""),
    ]
    for col, (label, val, delta) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Revenue Trend ─────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">📅 Monthly Revenue Trend</div>',
                    unsafe_allow_html=True)
        monthly = (
            df.groupby("year_month")["total_amount"]
            .sum()
            .reset_index()
            .rename(columns={"total_amount": "revenue"})
        )
        fig = px.area(
            monthly, x="year_month", y="revenue",
            title="", color_discrete_sequence=["#58A6FF"],
            labels={"year_month": "Month", "revenue": "Revenue (£)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(line_width=2.5, fill="tozeroy",
                          fillcolor="rgba(88,166,255,0.15)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🌍 Revenue by Country</div>',
                    unsafe_allow_html=True)
        country_rev = (
            df.groupby("country")["total_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig2 = px.bar(
            country_rev, x="total_amount", y="country",
            orientation="h", color="total_amount",
            color_continuous_scale="Blues",
            labels={"total_amount": "Revenue (£)", "country": ""},
        )
        fig2.update_layout(**PLOTLY_LAYOUT,
                           coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed",
                                      gridcolor="#21262D",
                                      linecolor="#30363D"))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Day-of-week & Hour heatmap ────────────────
    st.markdown('<div class="section-header">🕐 Purchase Behaviour Heatmap</div>',
                unsafe_allow_html=True)
    hm = (
        df.groupby(["day_of_week", "hour"])["total_amount"]
        .sum()
        .unstack(fill_value=0)
    )
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig3 = go.Figure(go.Heatmap(
        z      = hm.values,
        x      = [f"{h:02d}:00" for h in hm.columns],
        y      = [day_labels[d] for d in hm.index],
        colorscale = "Blues",
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Revenue: £%{z:,.0f}<extra></extra>",
    ))
    fig3.update_layout(**PLOTLY_LAYOUT,
                       xaxis_title="Hour of Day",
                       yaxis_title="Day of Week",
                       height=280)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 – RFM ANALYSIS
# ═══════════════════════════════════════════════════════

def page_rfm(rfm: pd.DataFrame):
    st.markdown("## 📊 RFM Analysis")
    st.markdown("Customer scoring across Recency, Frequency, and Monetary dimensions.")

    col1, col2 = st.columns(2)

    # Segment Donut
    with col1:
        st.markdown('<div class="section-header">🎯 Customer Segments</div>',
                    unsafe_allow_html=True)
        seg_counts = rfm["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        colors = [SEGMENT_COLORS.get(s, "#ADB5BD") for s in seg_counts["segment"]]
        fig = go.Figure(go.Pie(
            labels    = seg_counts["segment"],
            values    = seg_counts["count"],
            hole      = 0.55,
            marker_colors = colors,
            textinfo  = "percent+label",
            hovertemplate = "%{label}<br>Customers: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            showlegend = False,
            annotations=[dict(text=f"<b>{len(rfm):,}</b><br>Customers",
                              x=0.5, y=0.5, font_size=14,
                              font_color="#E6EDF3",
                              showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    # RFM Score Distribution
    with col2:
        st.markdown('<div class="section-header">📉 RFM Score Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = make_subplots(rows=1, cols=3,
                              subplot_titles=["Recency Score", "Frequency Score", "Monetary Score"])
        for i, col in enumerate(["r_score", "f_score", "m_score"], 1):
            counts = rfm[col].value_counts().sort_index()
            colors_bar = ["#E63946", "#F4A261", "#E9C46A", "#2A9D8F", "#457B9D"]
            fig2.add_trace(
                go.Bar(x=counts.index, y=counts.values,
                       marker_color=colors_bar[:len(counts)],
                       showlegend=False,
                       hovertemplate="Score: %{x}<br>Count: %{y:,}<extra></extra>"),
                row=1, col=i
            )
        fig2.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # RFM Scatter Matrix
    st.markdown('<div class="section-header">🔍 RFM Feature Relationships</div>',
                unsafe_allow_html=True)
    sample = rfm.sample(min(2000, len(rfm)), random_state=42)
    fig3 = px.scatter_3d(
        sample,
        x="recency", y="frequency", z="monetary",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.7,
        size_max=6,
        labels={"recency": "Recency (days)",
                "frequency": "Frequency",
                "monetary": "Monetary (£)"},
        title="3D RFM Space by Segment",
    )
    fig3.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # Segment KPI table
    st.markdown('<div class="section-header">📋 Segment Summary Table</div>',
                unsafe_allow_html=True)
    from src.rfm import segment_summary
    summary_df = segment_summary(rfm)
    summary_df["churn_rate"] = summary_df["churn_rate"].apply(lambda x: f"{x}%")
    summary_df["avg_monetary"] = summary_df["avg_monetary"].apply(lambda x: f"£{x:,.2f}")
    summary_df["avg_recency"]  = summary_df["avg_recency"].apply(lambda x: f"{x:.1f}d")
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "segment":        st.column_config.TextColumn("Segment"),
            "customer_count": st.column_config.NumberColumn("Customers", format="%d"),
            "avg_recency":    st.column_config.TextColumn("Avg Recency"),
            "avg_frequency":  st.column_config.NumberColumn("Avg Frequency"),
            "avg_monetary":   st.column_config.TextColumn("Avg Revenue"),
            "churn_rate":     st.column_config.TextColumn("Churn Rate"),
        }
    )


# ═══════════════════════════════════════════════════════
# PAGE 3 – CLUSTER EXPLORER
# ═══════════════════════════════════════════════════════

def page_clusters(rfm: pd.DataFrame):
    st.markdown("## 🔵 Cluster Explorer")
    st.markdown("Deep-dive into K-Means customer clusters.")

    if "cluster_name" not in rfm.columns:
        st.warning("⚠️ Run clustering pipeline first. Using segment labels instead.")
        rfm["cluster_name"] = rfm["segment"]

    # Sidebar cluster filter
    cluster_options = ["All"] + sorted(rfm["cluster_name"].unique().tolist())
    selected = st.selectbox("Select Cluster", cluster_options)

    if selected != "All":
        view = rfm[rfm["cluster_name"] == selected]
    else:
        view = rfm

    # ── Cluster Overview KPIs ────────────────────
    cols = st.columns(4)
    kpis = [
        ("👥 Customers",     f"{len(view):,}"),
        ("📅 Avg Recency",   f"{view['recency'].mean():.1f}d"),
        ("🔄 Avg Frequency", f"{view['frequency'].mean():.1f}"),
        ("💰 Avg Revenue",   f"£{view['monetary'].mean():,.2f}"),
    ]
    for col, (label, val) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.6rem">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📦 Cluster Size Distribution</div>',
                    unsafe_allow_html=True)
        cluster_counts = rfm["cluster_name"].value_counts().reset_index()
        cluster_counts.columns = ["cluster", "count"]
        fig = px.bar(
            cluster_counts, x="cluster", y="count",
            color="cluster",
            color_discrete_sequence=["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
                                     "#F4A261", "#A8DADC", "#6D6875"],
            text="count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                          xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💸 Revenue per Cluster</div>',
                    unsafe_allow_html=True)
        rev_cluster = (
            rfm.groupby("cluster_name")["monetary"]
            .sum()
            .reset_index()
            .sort_values("monetary", ascending=False)
        )
        fig2 = px.pie(
            rev_cluster, values="monetary", names="cluster_name",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig2.update_layout(**PLOTLY_LAYOUT, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Radar / Spider Chart (cluster profile) ────
    st.markdown('<div class="section-header">🕸 Cluster Profile Radar</div>',
                unsafe_allow_html=True)

    profile = rfm.groupby("cluster_name")[
        ["r_score", "f_score", "m_score", "recency", "frequency"]
    ].mean().round(2)

    cats = ["R Score", "F Score", "M Score"]
    radar_data = []
    for cluster, row in profile.iterrows():
        vals = [row["r_score"], row["f_score"], row["m_score"]]
        vals.append(vals[0])  # close polygon
        radar_data.append(go.Scatterpolar(
            r    = vals,
            theta= cats + [cats[0]],
            fill = "toself",
            name = cluster,
            opacity = 0.7,
        ))

    fig3 = go.Figure(radar_data)
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(28,33,40,0.8)",
            radialaxis=dict(visible=True, range=[0, 5],
                           gridcolor="#30363D", color="#8B949E"),
            angularaxis=dict(gridcolor="#30363D", color="#8B949E"),
        ),
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Filtered table
    st.markdown(f'<div class="section-header">🔎 Customer Table — {selected}</div>',
                unsafe_allow_html=True)
    display_cols = ["customer_id", "recency", "frequency", "monetary",
                    "r_score", "f_score", "m_score", "rfm_score",
                    "segment", "cluster_name", "churn"]
    display_cols = [c for c in display_cols if c in view.columns]
    st.dataframe(
        view[display_cols].sort_values("monetary", ascending=False).head(200),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════
# PAGE 4 – CHURN PREDICTOR
# ═══════════════════════════════════════════════════════

def page_churn(rfm: pd.DataFrame, bundle: dict):
    st.markdown("## 🔮 Churn Predictor")
    st.markdown("Real-time churn probability scoring for individual customers.")

    tab1, tab2 = st.tabs(["👤 Single Customer", "📋 Batch Scoring"])

    # ── Tab 1 : Single customer ───────────────────
    with tab1:
        st.markdown("### Enter Customer RFM Values")
        col1, col2, col3 = st.columns(3)

        with col1:
            recency   = st.slider("Recency (days since last purchase)", 1, 365, 60)
        with col2:
            frequency = st.slider("Frequency (number of orders)", 1, 100, 10)
        with col3:
            monetary  = st.slider("Monetary (total spend £)", 10, 20000, 500)

        if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
            import numpy as np
            row = pd.DataFrame([{
                "customer_id":    0,
                "recency":        recency,
                "frequency":      frequency,
                "monetary":       monetary,
                "r_score":        5 - min(4, recency // 73),
                "f_score":        min(5, max(1, frequency // 5)),
                "m_score":        min(5, max(1, int(monetary / 500))),
                "rfm_score":      0.0,
                "log_recency":    np.log1p(recency),
                "log_frequency":  np.log1p(frequency),
                "log_monetary":   np.log1p(monetary),
            }])
            row["rfm_score"] = (row["r_score"] + row["f_score"] + row["m_score"]) / 3

            from src.model import predict_churn
            result = predict_churn(row, bundle=bundle)
            prob   = result["churn_prob"].iloc[0]
            label  = result["risk_label"].iloc[0]

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                title = dict(text="Churn Probability (%)", font=dict(color="#E6EDF3")),
                delta = dict(reference=50, valueformat=".1f"),
                gauge = dict(
                    axis  = dict(range=[0, 100], tickcolor="#8B949E",
                                  tickfont=dict(color="#8B949E")),
                    bar   = dict(color="#E63946" if prob > 0.6 else
                                        "#F9C74F" if prob > 0.3 else "#2A9D8F"),
                    steps = [
                        dict(range=[0, 30],  color="rgba(42,157,143,0.15)"),
                        dict(range=[30, 60], color="rgba(249,199,79,0.15)"),
                        dict(range=[60, 100],color="rgba(230,57,70,0.15)"),
                    ],
                    threshold=dict(line=dict(color="#FFFFFF", width=3), value=50),
                ),
                number=dict(suffix="%", font=dict(color="#E6EDF3", size=48)),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3"),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk label card
            colour = "#E63946" if prob > 0.6 else "#F9C74F" if prob > 0.3 else "#2A9D8F"
            st.markdown(f"""
            <div style="background: rgba(28,33,40,0.9); border: 2px solid {colour};
                        border-radius: 12px; padding: 20px; text-align: center;">
              <h2 style="color: {colour}; margin: 0;">{label}</h2>
              <p style="color:#8B949E; margin: 8px 0 0 0;">
                {'This customer has a HIGH probability of churning. Immediate retention action recommended!'
                   if prob > 0.6 else
                 'This customer shows medium-risk signals. Consider targeted re-engagement.'
                   if prob > 0.3 else
                 'This customer appears healthy. Continue standard engagement programs.'}
              </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2 : Batch scoring ─────────────────────
    with tab2:
        st.markdown("### Batch Churn Scoring for All Customers")

        if st.button("🚀 Score All Customers", type="primary"):
            from src.model import predict_churn
            with st.spinner("Scoring customers…"):
                scores = predict_churn(rfm, bundle=bundle)
                scores = scores.merge(
                    rfm[["customer_id", "recency", "frequency", "monetary",
                          "segment"]],
                    on="customer_id", how="left"
                )

            st.success(f"✅ Scored {len(scores):,} customers")

            # Distribution
            fig = px.histogram(
                scores, x="churn_prob", nbins=40,
                color_discrete_sequence=["#E63946"],
                labels={"churn_prob": "Churn Probability"},
                title="Distribution of Churn Probabilities",
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            # Risk breakdown
            risk_counts = scores["risk_label"].value_counts().reset_index()
            risk_counts.columns = ["risk", "count"]
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.pie(risk_counts, values="count", names="risk",
                              color="risk",
                              color_discrete_map={
                                  "🔴 High Risk":   "#E63946",
                                  "🟡 Medium Risk": "#F9C74F",
                                  "🟢 Low Risk":    "#2A9D8F",
                              },
                              title="Customer Risk Distribution")
                fig2.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                high_risk = scores[scores["risk_label"] == "🔴 High Risk"]
                st.markdown(f"""
                <div class="metric-card" style="margin-top:30px;">
                  <div class="metric-value" style="color:#E63946;">
                      {len(high_risk):,}
                  </div>
                  <div class="metric-label">🔴 High Risk Customers</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                st.markdown("**Top 20 High-Risk Customers:**")
                st.dataframe(
                    high_risk[["customer_id", "churn_prob", "recency",
                                "frequency", "monetary", "segment"]]
                    .sort_values("churn_prob", ascending=False)
                    .head(20),
                    use_container_width=True,
                    hide_index=True,
                )

            # Download button
            csv = scores.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Scored CSV",
                data       = csv,
                file_name  = "churn_scores.csv",
                mime       = "text/csv",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════
# PAGE 5 – BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════

def page_insights(df: pd.DataFrame, rfm: pd.DataFrame):
    st.markdown("## 📈 Business Insights")

    tab1, tab2, tab3 = st.tabs(
        ["🔄 Cohort Analysis", "🛍 Product Performance", "🌍 Geographic"]
    )

    # ── Cohort Analysis ───────────────────────────
    with tab1:
        st.markdown("### Monthly Cohort Retention")

        # Build cohort table
        df2 = df.copy()
        df2["cohort_month"] = (
            df2.groupby("customer_id")["invoice_date"]
            .transform("min")
            .dt.to_period("M")
        )
        df2["order_month"]  = df2["invoice_date"].dt.to_period("M")
        df2["cohort_index"] = (
            (df2["order_month"] - df2["cohort_month"])
            .apply(lambda x: x.n)
        )

        cohort_data = (
            df2.groupby(["cohort_month", "cohort_index"])["customer_id"]
            .nunique()
            .reset_index()
            .rename(columns={"customer_id": "n_customers"})
        )

        cohort_pivot = cohort_data.pivot(
            index="cohort_month", columns="cohort_index", values="n_customers"
        )
        cohort_pct = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0).round(3)
        cohort_pct = cohort_pct.iloc[:, :12]   # first 12 months

        fig = go.Figure(go.Heatmap(
            z          = cohort_pct.values * 100,
            x          = [f"Month {i}" for i in cohort_pct.columns],
            y          = [str(c) for c in cohort_pct.index],
            colorscale = "Blues",
            text       = np.round(cohort_pct.values * 100, 1),
            texttemplate="%{text}%",
            hovertemplate="Cohort: %{y}<br>%{x}<br>Retention: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          xaxis_title="Cohort Period",
                          yaxis_title="Cohort Month",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Product Performance ───────────────────────
    with tab2:
        st.markdown("### Top Products by Revenue")

        top_products = (
            df.groupby(["stock_code", "description"])
            .agg(
                total_revenue = ("total_amount", "sum"),
                total_qty     = ("quantity",     "sum"),
                orders        = ("invoice_no",   "nunique"),
            )
            .reset_index()
            .sort_values("total_revenue", ascending=False)
            .head(20)
        )
        top_products["description"] = top_products["description"].str[:35]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                top_products.head(10),
                x="total_revenue", y="description",
                orientation="h", color="total_revenue",
                color_continuous_scale="Blues",
                title="Top 10 Products by Revenue",
                labels={"total_revenue": "Revenue (£)", "description": ""},
            )
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed",
                                         gridcolor="#21262D",
                                         linecolor="#30363D"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.treemap(
                top_products,
                path    = ["description"],
                values  = "total_revenue",
                color   = "total_qty",
                color_continuous_scale="Blues",
                title   = "Product Revenue Treemap",
            )
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            top_products.style.background_gradient(subset=["total_revenue"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )

    # ── Geographic ────────────────────────────────
    with tab3:
        st.markdown("### Revenue by Geography")

        country_data = (
            df.groupby("country")
            .agg(
                revenue   = ("total_amount",  "sum"),
                customers = ("customer_id",   "nunique"),
                orders    = ("invoice_no",    "nunique"),
            )
            .reset_index()
            .sort_values("revenue", ascending=False)
        )

        fig = px.choropleth(
            country_data,
            locations          = "country",
            locationmode       = "country names",
            color              = "revenue",
            hover_data         = ["customers", "orders"],
            color_continuous_scale="Blues",
            title              = "Revenue by Country",
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            geo=dict(bgcolor="rgba(0,0,0,0)",
                     showframe=False,
                     showcoastlines=True,
                     coastlinecolor="#30363D",
                     landcolor="#21262D",
                     oceancolor="#0D1117",
                     showocean=True),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# MAIN  – App entry point
# ═══════════════════════════════════════════════════════

def main():
    page, data_path = sidebar()

    # ── Check data exists ────────────────────────
    if not os.path.exists(data_path):
        st.error(f"""
        ❌ **Data file not found:** `{data_path}`

        Please ensure the Online Retail II CSV file is placed at the path above,
        or update the path in the sidebar.
        """)
        st.stop()

    # ── Load data ────────────────────────────────
    try:
        df, rfm = load_all_data(data_path)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

    # ── Route to page ────────────────────────────
    if page == "🏠 Overview":
        page_overview(df, rfm)

    elif page == "📊 RFM Analysis":
        page_rfm(rfm)

    elif page == "🔵 Cluster Explorer":
        page_clusters(rfm)

    elif page == "🔮 Churn Predictor":
        bundle = load_or_train_model(rfm)
        page_churn(rfm, bundle)

    elif page == "📈 Business Insights":
        page_insights(df, rfm)


if __name__ == "__main__":
    main()