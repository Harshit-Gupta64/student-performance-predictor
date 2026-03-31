import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

#load dataset
def load_student_data() -> pd.DataFrame:
	base_dir = Path(__file__).resolve().parent #get the folder where this script is located
	data_path = base_dir.parent / "data" / "student_data.csv" #goes up one level, then into the data folder, and then to the student_data.csv file

	try:
		return pd.read_csv(data_path)
	except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
		# Fallback for files saved in Excel format but with a .csv extension.
		return pd.read_excel(data_path)


#dataset exploration
df = load_student_data()

# features & target
X = df[["study_hours", "attendance", "sleep_hours"]]
y = df["marks"]

# train model
model = LinearRegression()
model.fit(X, y)

st.title("Student Marks Predictor")

# user inputs
study_hours = st.slider("Study Hours", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
sleep_hours = st.slider("Sleep Hours", 0, 10, 7)

# prediction
if st.button("Predict"):
    prediction = model.predict([[study_hours, attendance, sleep_hours]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")