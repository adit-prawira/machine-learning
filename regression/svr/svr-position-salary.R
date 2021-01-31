dataset = read.csv("Position_Salaries.csv")[2:3]

library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = "eps-regression")
#Predicting Salary of position level 
position_level = 6.5
y_pred = predict(regressor, data.frame(Level = position_level))


# Visualize SVR prediction model
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, 
                 y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = dataset$Level, 
                y = predict(regressor, newdata = dataset)),
            color = "blue") +
  ggtitle("Truth or Bluff (SVR Regression)") +
  xlab("Position Level") + 
  ylab("Salary")