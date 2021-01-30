# REMAINDER: index in R starts at 1 not 0
# Only takes values of levels each of its salary
dataset = read.csv("Position_Salaries.csv")[2:3]

# Fit Linear regression model
lin_reg = lm(formula = Salary ~ ., 
             data = dataset)

# Fit polynomial regression

# Second Order dependency
dataset$Level2 = dataset$Level^2

# Third Order dependency
dataset$Level3 = dataset$Level^3
# Fourth Order dependency
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~ .,
              data = dataset)


# Visualizing Linear Regression Result
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, 
                 y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = dataset$Level, 
                y = predict(lin_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Truth or Bluff (Linear Regression)") +
  xlab("Position Level") + 
  ylab("Salary")

# Visualizing Polynomial Regression Result
ggplot() + 
  geom_point(aes(x = dataset$Level, 
                 y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = dataset$Level, 
                y = predict(poly_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Truth or Bluff (Linear Regression)") +
  xlab("Position Level") + 
  ylab("Salary")

# Predicting Salary of position level (Linear regression)
y_pred_lin = predict(lin_reg, data.frame(Level = 6.5))

# Predicting Salary of position level (Polynomial regression)
y_pred_poly = predict(poly_reg, data.frame(Level = 6.5,
                                           Level2 = 6.5^2,
                                           Level3 = 6.5^3,
                                           Level4 = 6.5^4))