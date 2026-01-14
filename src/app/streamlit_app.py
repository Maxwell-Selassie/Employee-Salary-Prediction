from __future__ import annotations

"""
Streamlit demo app for business users.

Features:
- Manual feature input for single prediction
- CSV upload for batch prediction
- SHAP-based feature contribution visualization
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from inference.service import InferenceService


st.set_page_config(page_title="Employee Salary Prediction", layout="wide")


@st.cache_resource
def get_inference_service() -> InferenceService:
    return InferenceService()


service = get_inference_service()

st.title("Employee Salary Prediction – Business Demo")

st.markdown(
    """
This application serves a production-grade regression model tracked and registered with MLflow.

**How it works**
- Data is validated and preprocessed using a reproducible pipeline
- Several models (Ridge, RandomForest, XGBoost, LightGBM) are trained
- The best model is selected via cross-validated MSE under a <10s runtime constraint
- The winning model is registered to MLflow and exposed via the `production` alias
"""
)

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch (CSV)"])

with tab_single:
    st.subheader("Manual Input")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Employee Age", min_value=18, max_value=80, value=30)
        years_exp = st.number_input("Years Experience", min_value=0, max_value=40, value=5)
    with col2:
        n_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        dept = st.text_input("Department", "Engineering")
    with col3:
        role = st.text_input("Role", "Software Engineer")
        perf = st.slider("Performance Rating", min_value=1, max_value=5, value=3)

    if st.button("Predict Salary"):
        features = {
            "Employee_age": age,
            "years_experience": years_exp,
            "Number_of_Children": n_children,
            "Department": dept,
            "Role": role,
            "performance_rating": perf,
        }
        pred = service.predict_single(features)
        st.metric("Predicted Salary", f"${pred:,.2f}")

        st.subheader("Feature Contributions (SHAP)")
        shap_values, X = service.explain_shap(pd.DataFrame([features]))
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.bar_plot(shap_values[0], feature_names=X.columns, show=False)
        st.pyplot(fig)

with tab_batch:
    st.subheader("Batch Prediction from CSV")
    uploaded = st.file_uploader("Upload CSV with input features", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        preds = service.predict_batch(df)
        df_out = df.copy()
        df_out["predicted_salary"] = preds
        st.write("Results:", df_out.head())

        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        st.download_button("Download Predictions", data=csv_buf.getvalue(), file_name="predictions.csv")

