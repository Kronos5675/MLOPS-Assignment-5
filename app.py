import streamlit as st
import joblib, numpy as np
from sklearn.datasets import load_diabetes

@st.cache_resource 
def load_resources():
    try:
        # Load the model and feature names for the interface
        return joblib.load('model/regression_model.pkl'), load_diabetes().feature_names
    except:
        return None, None

model, feature_names = load_resources()

st.title("MLOps Diabetes Score Predictor")
st.markdown("---")

if model is not None:
    st.subheader("Model Status: Ready (Linear Regression)")
    input_values = []
    
    # Create 10 input fields for the 10 features
    for name in feature_names:
        # Use a reasonable default value and step for numerical data
        val = st.number_input(f"Input for: {name}", value=0.0, step=0.001, format="%.4f")
        input_values.append(val)
        
    if st.button("Calculate Prediction"):
        try:
            # Prepare input array (1 row, 10 columns)
            prediction_score = model.predict(np.array(input_values).reshape(1, -1))[0]
            st.success(f"**Predicted Disease Progression Score:** **{prediction_score:.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.error("Model artifact failed to load.")
