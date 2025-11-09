# =======================================
# Advanced Cancer Prediction Dashboard
# =======================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from PIL import Image

# =======================================
# Page Config
# =======================================
st.set_page_config(
    page_title="Cancer Prediction System",
    page_icon="🩺",
    layout="wide"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cancer_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SUMMARY_PATH = os.path.join(MODEL_DIR, "summary.json")
LOG_PATH = os.path.join(MODEL_DIR, "model_logs.csv")

# =======================================
# Load model & scaler
# =======================================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("✅ Model & Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model/scaler: {e}")

# =======================================
# Sidebar Info
# =======================================
st.sidebar.title("🩺 Cancer Prediction System")
st.sidebar.markdown("A Machine Learning dashboard to analyze, compare and predict Breast Cancer outcomes.")
st.sidebar.markdown("---")
st.sidebar.info("Developed by **Mathew Titus**")

# =======================================
# Tabs
# =======================================
tabs = st.tabs([
    "🔮 Prediction",
    "📊 Model Comparison",
    "📈 Feature Insights",
    "📋 Summary Dashboard"
])

# =======================================
# TAB 1 - Prediction
# =======================================
with tabs[0]:
    st.header("🔮 Predict Cancer Outcome")

    # Input section (based on 5 features)
    st.write("Enter tumor mean characteristics:")

    col1, col2, col3 = st.columns(3)
    with col1:
        mean_radius = st.number_input("Mean Radius", min_value=0.0, step=0.01)
        mean_area = st.number_input("Mean Area", min_value=0.0, step=1.0)
    with col2:
        mean_texture = st.number_input("Mean Texture", min_value=0.0, step=0.01)
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, step=0.0001)
    with col3:
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, step=0.01)

    if st.button("🔍 Predict"):
        try:
            input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(scaled_input)[0][1]
            else:
                proba = None

            if prediction == 0:
                st.error("⚠️ Result: **Malignant (Cancerous Tumor)**")
            else:
                st.success("✅ Result: **Benign (Non-Cancerous Tumor)**")

            if proba is not None:
                st.progress(proba)
                st.caption(f"Prediction Confidence: {proba*100:.2f}% (Benign)")

        except Exception as e:
            st.error(f"Error: {e}")

# =======================================
# TAB 2 - Model Comparison
# =======================================
with tabs[1]:
    st.header("📊 Model Comparison Dashboard")

    try:
        df_logs = pd.read_csv(LOG_PATH)
        st.dataframe(df_logs, use_container_width=True)

        # Show bar charts
        st.subheader("Accuracy Comparison")
        st.bar_chart(df_logs.set_index("model")["accuracy"])

        st.subheader("AUC Comparison")
        st.bar_chart(df_logs.set_index("model")["auc"])

    except FileNotFoundError:
        st.warning("No model_logs.csv found. Run Model_Retrain_Advanced.ipynb first.")

# =======================================
# TAB 3 - Feature Insights
# =======================================
with tabs[2]:
    st.header("📈 Feature Insights and Model Visuals")

    try:
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists(os.path.join(MODEL_DIR, "roc_curves.png")):
                st.image(os.path.join(MODEL_DIR, "roc_curves.png"), caption="ROC Curves (Top Models)")
            else:
                st.warning("ROC curve not found.")

        with col2:
            if os.path.exists(os.path.join(MODEL_DIR, "feature_importance.png")):
                st.image(os.path.join(MODEL_DIR, "feature_importance.png"), caption="Feature Importance")
            else:
                st.info("Feature importance not available for this model.")

        st.divider()

        if os.path.exists(os.path.join(MODEL_DIR, "correlation_heatmap.png")):
            st.image(os.path.join(MODEL_DIR, "correlation_heatmap.png"), caption="Correlation Heatmap")

    except Exception as e:
        st.error(f"Error displaying insights: {e}")

# =======================================
# TAB 4 - Summary Dashboard
# =======================================
with tabs[3]:
    st.header("📋 Model Summary and Performance")

    try:
        # Summary JSON
        if os.path.exists(SUMMARY_PATH):
            with open(SUMMARY_PATH, "r") as f:
                summary = json.load(f)
            st.json(summary)

        # Classification Report
        report_path = os.path.join(MODEL_DIR, "report.txt")
        if os.path.exists(report_path):
            st.subheader("📄 Classification Report")
            with open(report_path, "r") as f:
                st.text(f.read())

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(os.path.join(MODEL_DIR, "accuracy_comparison.png")):
                st.image(os.path.join(MODEL_DIR, "accuracy_comparison.png"), caption="Accuracy Chart")
        with col2:
            if os.path.exists(os.path.join(MODEL_DIR, "auc_comparison.png")):
                st.image(os.path.join(MODEL_DIR, "auc_comparison.png"), caption="AUC Chart")

    except Exception as e:
        st.error(f"Error in summary tab: {e}")

# =======================================
# Footer
# =======================================
st.sidebar.markdown("---")
st.sidebar.markdown("📅 Project Version: Advanced v1.0")
st.sidebar.caption("Built using **Python, scikit-learn, and Streamlit**.")
