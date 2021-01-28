dataset = read.csv("50_Startups.csv")

dataset$State = factor(dataset$State, 
                       levels = c("New York", "California", "Florida"),
                       labels = c(1,2,3))

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# From regressor p value of R.D.Spend (p = 6.70e-16) is highly 
# statistically significant
regressor = lm(formula = Profit ~ ., data = training_set)

# Predciting the test set result of profits
y_pred = predict(regressor, newdata = test_set)
summary(regressor)
# Building Optimal model for Backward Elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    
    # find the all maximum p values after every elimination
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      
      # find the index of the data which has the maximum p values
      # and eliminate it (e.g. Marketing.Spend, Administration etc)
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    
    # Reduce data length by 1 to indicates elimination
    numVars = numVars - 1
  }
  return(summary(regressor))
}

# set significant level of 5%
SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)