import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(x,y)
y_pred = regressor.predict(x)

position_level = 6.5
salary_prediction = regressor.predict([[position_level]])[0]
print(f"\nPrediction with Random Forest Regression of {position_level} position level: ${salary_prediction}\n")

plt.scatter(x, y, color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(x, y_pred, color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
y_pred_high = regressor.predict(x_grid)
plt.scatter(x, y, color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(x_grid, y_pred_high, color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Higher Resolution Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()