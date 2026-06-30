import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Semiconductor Failure Prediction",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Semiconductor Process Failure Prediction")
st.markdown("**SECOM Dataset | Predictive Maintenance Dashboard**")
st.markdown("---")

# Load model artifacts
@st.cache_resource
def load_artifacts():
    with open('models/final_model.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts['model']

# Sidebar info
st.sidebar.header("📋 Model Info")
st.sidebar.markdown("""
**Model:** Logistic Regression
**Features:** 80 (selected via SelectKBest)
**Class balancing:** class_weight='balanced'

**Test Performance:**
- Recall (Fail): 62%
- Precision (Fail): 17%
- AUC-ROC: 0.685
""")

st.sidebar.markdown("---")
st.sidebar.header("📊 Simulation Controls")
n_samples = st.sidebar.slider("Number of wafers to simulate", 10, 100, 50)
threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

if st.sidebar.button("🔄 Run Prediction Batch", type="primary"):
    # Simulate incoming wafer sensor data (80 features sau feature selection)
    np.random.seed(int(np.random.rand() * 1000))
    sim_data = np.random.randn(n_samples, model.n_features_in_)

    probabilities = model.predict_proba(sim_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    col1, col2, col3, col4 = st.columns(4)
    fail_count = predictions.sum()
    col1.metric("Total Wafers", n_samples)
    col2.metric("Predicted Failures", int(fail_count))
    col3.metric("Yield Rate", f"{(1 - fail_count/n_samples)*100:.1f}%")
    col4.metric("Avg Risk Score", f"{probabilities.mean():.3f}")

    st.markdown("### Risk Distribution")
    fig = px.histogram(
        x=probabilities, nbins=20,
        labels={'x': 'Failure Probability'},
        color_discrete_sequence=['#EF553B']
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold={threshold}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Wafer-level Predictions")
    df_results = pd.DataFrame({
        'Wafer ID': [f'W{i:04d}' for i in range(n_samples)],
        'Failure Probability': probabilities.round(3),
        'Prediction': ['🔴 FAIL' if p == 1 else '🟢 PASS' for p in predictions],
        'Risk Level': ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.4 else 'LOW'
                       for p in probabilities]
    })
    st.dataframe(
        df_results.sort_values('Failure Probability', ascending=False),
        use_container_width=True, height=400
    )
else:
    st.info("👈 Bấm **Run Prediction Batch** ở sidebar để bắt đầu simulation")

    st.markdown("### 📖 About this project")
    st.markdown("""
    Dự án này xây dựng model dự đoán failure cho wafer trong quy trình
    sản xuất bán dẫn, sử dụng dataset SECOM (UCI ML Repository) gồm
    591 sensor readings từ 1,567 wafers thực tế.

    **Key challenges giải quyết:**
    - Severe class imbalance (93% pass / 7% fail)
    - High dimensionality (590 sensors → reduced to 80 via feature selection)
    - Missing data (~4.5% sensor readings)

    **Technical approach:**
    - So sánh SMOTE vs class_weight balancing — class_weight thắng trên
      dataset có quá ít minority samples
    - So sánh XGBoost, Random Forest, Logistic Regression — model tuyến
      tính đơn giản generalize tốt hơn do overfit risk với 296 features
      ban đầu
    - Tối ưu cho **recall** thay vì precision, vì cost của một undetected
      failure trong fab cao hơn nhiều so với false alarm
    """)