import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# Reshape y into a 2D array

y = y.reshape(len(y),1)
# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset

regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)
# Predict the result rescaled sc_x

y_pred = regressor.predict(x)

# Predicting Salary of position level 
position_level = 6.5
pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))[0]
print(f"\nPrediction with SVR Regression of {position_level} position level: ${pred}\n")

# Visualize SVR prediction model by converting x and y to its original form/dimension
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(y_pred), color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Higher resolution SVR regression model
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
y_pred_high = regressor.predict(x_grid)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(y_pred_high), color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Higher Resolution SVR Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()