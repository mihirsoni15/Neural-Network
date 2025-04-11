import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="🎓 UCLA Admission Predictor", layout="centered")
st.title("🎓 Neural Network Admission Predictor")

# Load trained model
model_path = 'models/admission_nn_model.pkl'
if not os.path.exists(model_path):
    st.error("❌ Model not found. Please train the model using `train.py`.")
    st.stop()

model = joblib.load(model_path)

# Input form
st.subheader("📋 Enter your academic profile:")

gre = st.slider("GRE Score", 260, 340, 310)
toefl = st.slider("TOEFL Score", 0, 120, 100)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0, step=0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0, step=0.5)
cgpa = st.slider("CGPA (out of 10)", 6.0, 10.0, 8.5, step=0.1)
research = st.radio("Research Experience", ["Yes", "No"]) == "Yes"

# One-hot encode University_Rating and Research
input_dict = {
    "GRE_Score": gre,
    "TOEFL_Score": toefl,
    "SOP": sop,
    "LOR": lor,
    "CGPA": cgpa,
    "University_Rating_2": 1 if rating == 2 else 0,
    "University_Rating_3": 1 if rating == 3 else 0,
    "University_Rating_4": 1 if rating == 4 else 0,
    "University_Rating_5": 1 if rating == 5 else 0,
    "Research_1": 1 if research else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure feature alignment
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("🔮 Predict Admission"):
    prediction = model.predict(input_df)[0]
    label = "🎉 High Chance of Admission!" if prediction == 1 else "❌ Low Chance of Admission"

    st.subheader("📢 Prediction Result:")
    st.markdown(f"### {label}")
    
    st.subheader("📈 Feature Overview:")
    st.bar_chart(input_df.T.rename(columns={0: 'Value'}))
