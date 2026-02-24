import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(page_title="Customer Retention Dashboard", page_icon="📊", layout="wide")

# Header Section
st.title("📊 Customer Segmentation & Retention Dashboard")
st.markdown("Identify high-value customers and predict flight risks using Machine Learning.")
st.markdown("---")

# Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/rfm_clusters.csv")

@st.cache_resource
def load_model():
    return joblib.load("reports/churn_xgb_model.pkl")

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files. Please make sure Phase 3 and Phase 4 are completed. Details: {e}")
    st.stop()

# Sidebar: Customer Search
st.sidebar.header("🔍 Find Customer")
st.sidebar.markdown("Select a customer ID to view their profile and churn risk.")
customer_list = df['CustomerID'].astype(str).tolist()
selected_id = st.sidebar.selectbox("Customer ID", customer_list)

# Main Dashboard 
if selected_id:
    # Get specific customer data
    customer_data = df[df['CustomerID'].astype(str) == selected_id].iloc[0]
    
    st.header(f"Profile: Customer #{selected_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recency", f"{customer_data['Recency']} Days")
    col2.metric("Frequency", f"{customer_data['Frequency']} Orders")
    col3.metric("Monetary", f"${customer_data['Monetary']:,.2f}")
    
    segment = customer_data['Customer_Persona']
    if segment == 'Champions':
        col4.success(f"🏆 {segment}")
    elif segment == 'At-Risk':
        col4.error(f"⚠️ {segment}")
    else:
        col4.info(f"👥 {segment}")
        
    st.markdown("---")
    
    st.subheader("🔮 AI Churn Prediction")
    
    # Prepare features for the XGBoost model
    features = pd.DataFrame({
        'Frequency': [customer_data['Frequency']], 
        'Monetary': [customer_data['Monetary']]
    })
    
    # Predict Probability
    churn_prob = model.predict_proba(features)[0][1] * 100
    
    col_pred, col_advice = st.columns(2)
    
    with col_pred:
        st.markdown(f"**Probability of Leaving:** {churn_prob:.1f}%")
        st.progress(int(churn_prob))
        
        if churn_prob > 50:
            st.error("High Flight Risk: This customer is likely to churn.")
        else:
            st.success("Safe: This customer is likely to be retained.")
            
    with col_advice:
        st.markdown("💡 **Recommended Business Action:**")
        if segment == 'Champions':
            st.write("- Reward them with VIP access or loyalty points to maintain engagement.")
        elif segment == 'At-Risk':
            st.write("- Send a highly personalized 'We miss you' email with a heavy discount.")
        elif segment == 'Lost / Hibernating':
            st.write("- Do not allocate high marketing budget here. Send standard promotional emails.")
        else:
            st.write("- Engage with standard upselling campaigns to boost frequency.")