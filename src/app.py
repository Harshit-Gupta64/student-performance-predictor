import streamlit as st
from main import train_model

model = train_model()

st.title("Student Marks Predictor")

# user inputs
weekly_self_study_hours = st.slider("Weekly Self-Study Hours", 0, 40, 20)
attendance_percentage = st.slider("Attendance (%)", 0, 100, 70)
class_participation = st.slider("Class Participation", 0, 10, 3)

# prediction
if st.button("Predict"):
    prediction = model.predict([[weekly_self_study_hours, attendance_percentage, class_participation]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")