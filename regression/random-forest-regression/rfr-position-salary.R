dataset = read.csv("Position_Salaries.csv")[2:3]

library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)
#Predicting Salary of position level 
position_level = 6.5
y_pred = predict(regressor, data.frame(Level = position_level))


# Visualize SVR prediction model
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, 
                 y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = x_grid, 
                y = predict(regressor, data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Truth or Bluff (SVR Regression)") +
  xlab("Position Level") + 
  ylab("Salary")