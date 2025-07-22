import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("student_score_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎓 Student Exam Score Predictor")
st.markdown("""
Welcome to the **Student Performance Prediction App**.
Provide the student's input data below to estimate their expected exam score.
""")

# Sidebar inputs
st.sidebar.header("📋 Input Student Details")

hours = st.sidebar.slider("📚 Hours Studied", 0.0, 24.0, 2.0, 0.5)
previous_score = st.sidebar.slider("📝 Previous Exam Score", 0.0, 100.0, 70.0, 1.0)
attendance = st.sidebar.slider("📅 Attendance (%)", 0.0, 100.0, 90.0, 1.0)

# Input DataFrame
input_df = pd.DataFrame([[hours, previous_score, attendance]],
                        columns=["Hours_Studied", "Previous_Score", "Attendance"])

# Prediction
if st.sidebar.button("🎯 Predict Score"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    st.subheader("📊 Prediction Result")
    st.success(f"🎓 The predicted exam score is **{prediction:.2f}** out of 100.")

# Footer
st.markdown("""
---
Made with ❤️ using Sarthak  
""")
