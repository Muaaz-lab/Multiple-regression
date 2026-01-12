import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [50,55,60,65,70,75,80,85,90,95],
    "Previous_Score": [30,35,40,45,50,60,65,70,80,90],
    "Final_Marks": [35,38,42,45,50,60,65,70,80,90]
}

df = pd.DataFrame(data)
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Marks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
new_student = [[6, 78, 62]]  # Study Hours, Attendance, Previous Score
predicted_marks = model.predict(new_student)
mse = mean_squared_error(y_test, y_pred)

print("MSE (Mean Squared Error):", mse)


print("Predicted Final Marks:", round(predicted_marks[0], 2))
