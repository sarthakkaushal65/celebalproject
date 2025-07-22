import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and training features
model = joblib.load("student_score_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")  # This is a list of columns used during training

# Title and layout
st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("📘 Student Exam Score Predictor")
st.markdown("Predicts student's exam score based on various factors.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
hours_studied = st.slider("Hours Studied", 0.0, 12.0, 3.0)
previous_score = st.slider("Previous Exam Score", 0, 100, 70)
attendance = st.slider("Attendance (%)", 0, 100, 90)
internet = st.selectbox("Internet Access", ["Yes", "No"])
extra_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
health_status = st.selectbox("Health Status", ["Poor", "Average", "Good"])
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
distance_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

# Prepare DataFrame
input_dict = {
    "Gender": gender,
    "Hours_Studied": hours_studied,
    "Previous_Exam_Score": previous_score,
    "Attendance": attendance,
    "Internet_Access": internet,
    "Extracurricular_Activities": extra_activities,
    "Health_Status": health_status,
    "Sleep_Hours": sleep,
    "Access_to_Resources": access_resources,
    "Distance_from_Home": distance_home
}

df = pd.DataFrame([input_dict])

# One-hot encoding
df_encoded = pd.get_dummies(df)

# Align with training features
df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

# Scale numerical features (optional depending on how you trained)
scaled_input = scaler.transform(df_encoded)

# Predict
if st.button("Predict"):
    score = model.predict(scaled_input)[0]
    st.success(f"📊 Predicted Exam Score: **{score:.2f}%**")
