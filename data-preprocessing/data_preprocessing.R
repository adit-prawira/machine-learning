
# import data set
dataset = read.csv("Data.csv")

# handle missing data, in which in this case are missing data within 
# Age and Salary
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Encoding categorical data
dataset$Country = factor(dataset$Country, 
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased, 
                         levels = c("No", "Yes"),
                         labels = c(0, 1))

# Splitting the dataset into training(80% observation) and 
# test(20% observation) set with random factor of 1
# install.packages("caTools)
library(caTools)
set.seed(123)

# this will return values of TRUEs and FALSEs where the TRUEs indicates
# train set and the FALSEs indicates test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling with Standardisation
# training_set[, 2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])