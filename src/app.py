import streamlit as st
from main import train_model

model = train_model()

st.title("Student Marks Predictor")

# user inputs
study_hours = st.slider("Study Hours", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
sleep_hours = st.slider("Sleep Hours", 0, 10, 7)

# prediction
if st.button("Predict"):
    prediction = model.predict([[study_hours, attendance, sleep_hours]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")