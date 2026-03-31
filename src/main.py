from pathlib import Path
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression



#load dataset
def load_student_data() -> pd.DataFrame:
	base_dir = Path(__file__).resolve().parent #get the folder where this script is located
	data_path = base_dir.parent / "data" / "student_performance.csv" #goes up one level, then into the data folder, and then to the student_data.csv file

	try:
		return pd.read_csv(data_path)
	except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
		# Fallback for files saved in Excel format but with a .csv extension.
		return pd.read_excel(data_path)


def train_model():
    
    df= load_student_data()
    X = df[["weekly_self_study_hours", "attendance_percentage", "class_participation"]]
    y = df["total_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions= model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    return model
