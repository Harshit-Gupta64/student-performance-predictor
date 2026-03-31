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


#dataset exploration
df = load_student_data()
print(df)

#information about the dataset
# print("\nFirst 5 rows:")
# print(df.head())

# print("\nData info:")
# df.info()

# print("\nSummary statistics:")
# print(df.describe())

#conditional filtering
# print("\nStudents with marks > 70:")
# high_marks=df[df["marks"]>70] #first df is the original dataframe, then we filter it to only include rows where the "marks" column is greater than 70
# print(high_marks)
# filtered = df[(df["marks"] > 70) & (df["attendance"] > 80)]
# print(filtered)
# print(df.iloc[0]["marks"]) #iloc is used to access a specific row and column by index. Here, we are accessing the first row (index 0) and the "marks" column.

# print("\nStudents sorted by marks (highest first):")
# sorted_df=df.sort_values(by="marks",ascending=False)
# print(sorted_df)

# print("\n Selected columns by (study_hours and marks):")
# selected=df[["study_hours","marks"]] 
# print(selected)

# print(" \n Average marks by study hours: ")
# print(df.groupby("study_hours")["marks"].mean())


#Plotting Graph
# print("\n Plotting PLot:")

# plt.scatter(df["study_hours"],df["marks"]) #plots point

# #names axes and title
# plt.xlabel("Study Hours")
# plt.ylabel("Marks")
# plt.title("Study Hours vs Marks")

# plt.show() #displays graph

X = df[["study_hours", "attendance", "sleep_hours"]]
y = df["marks"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

# train_test_split
# training (learn)
# testing (evaluate)

# test_size=0.2
# 20% test, 80% train

# random_state=42
# ensures same split every time

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))

model =LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test) # model prediccts the answer for the test data, which is stored in the predictions variable

print("\nPredictions:", predictions) #model predict
print("Actual:", list(y_test)) #what was actually in the test data

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\nMean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# new student data (example)
new_data = [[6, 80, 7]]  # study_hours, attendance, sleep_hours

prediction = model.predict(new_data)

print("\nPredicted marks for new student:", prediction[0])