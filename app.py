import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("student_score_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Student Exam Score Predictor",
    layout="centered"
)

st.title("📚 Student Exam Score Predictor")
st.markdown("This app predicts a student's exam score based on various academic and personal factors.")

# Input Fields – using at least 10 features
hours_studied = st.slider("Hours Studied per Day", 0.0, 12.0, 2.0, step=0.1)
previous_exam_score = st.slider("Previous Exam Score (%)", 0, 100, 70)
attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
study_materials_used = st.selectbox("Study Materials Used", ["Books", "Online", "Both", "None"])
extra_classes_attended = st.selectbox("Extra Classes Attended", ["Yes", "No"])
parental_education = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
part_time_job = st.selectbox("Has a Part-time Job?", ["Yes", "No"])
internet_access = st.selectbox("Internet Access at Home?", ["Yes", "No"])
daily_sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.0, step=0.1)
self_study_hours = st.slider("Daily Self-Study Hours", 0.0, 10.0, 2.0, step=0.5)

# Encode categorical fields
study_materials_map = {"Books": 0, "Online": 1, "Both": 2, "None": 3}
extra_classes_map = {"Yes": 1, "No": 0}
parental_education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
part_time_job_map = {"Yes": 1, "No": 0}
internet_access_map = {"Yes": 1, "No": 0}

# Prepare input
input_data = pd.DataFrame([[
    hours_studied,
    previous_exam_score,
    attendance_rate,
    study_materials_map[study_materials_used],
    extra_classes_map[extra_classes_attended],
    parental_education_map[parental_education],
    part_time_job_map[part_time_job],
    internet_access_map[internet_access],
    daily_sleep_hours,
    self_study_hours
]], columns=[
    "Hours_Studied",
    "Previous_Exam_Score",
    "Attendance",
    "Study_Materials_Used",
    "Extra_Classes_Attended",
    "Parental_Education_Level",
    "Part_Time_Job",
    "Internet_Access",
    "Daily_Sleep_Hours",
    "Self_Study_Hours"
])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Exam Score"):
    predicted_score = model.predict(scaled_input)[0]
    st.success(f"🎯 Predicted Exam Score: **{predicted_score:.2f}%**")
