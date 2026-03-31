from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error



#load dataset
def load_student_data() -> pd.DataFrame:
	base_dir = Path(__file__).resolve().parent #get the folder where this script is located
	data_path = base_dir.parent / "data" / "student_data.csv" #goes up one level, then into the data folder, and then to the student_data.csv file

	try:
		return pd.read_csv(data_path)
	except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
		# Fallback for files saved in Excel format but with a .csv extension.
		return pd.read_excel(data_path)



def train_model():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    df = load_student_data()

    X = df[["study_hours", "attendance", "sleep_hours"]]
    y = df["marks"]

    model = LinearRegression()
    model.fit(X, y)

    return model