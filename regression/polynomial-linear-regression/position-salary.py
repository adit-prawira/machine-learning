import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(x, y)

# construct polynomial features y = b0 + b1x + b2x^2
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x)

# Polynomial linear regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Higher degree polynomial linear regression
poly_reg_4 = PolynomialFeatures(degree=4)
X_poly_4 = poly_reg_4.fit_transform(x)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly_4, y)

# Linear regression 
plt.scatter(x, y, color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(x, lin_reg.predict(x), color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Polynomial Linear regression 
plt.scatter(x, y, color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(x, lin_reg_2.predict(X_poly), color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Higher Order Polynomial Linear regression 
plt.scatter(x, y, color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(x, lin_reg_4.predict(X_poly_4), color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Higher Order Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Higher resolution polynomial regression model
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color="red") # the scatter point of level (x) with the real salary(y)
plt.plot(x_grid, lin_reg_4.predict(poly_reg_4.fit_transform(x_grid)), color="blue")  # predicted linear line of level with the predicted salary(lin_reg.predict)
plt.title("Truth or Bluff (Higher Resolution Higher Order Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()