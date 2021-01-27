import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values

# Handle missing data and insert the mean value of all existing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding the independent variable, this will label country in each row that contains values of
# Age, and Salary with 1.0 and convert the other country with 0.0
# Country,Age,Salary
# France,44,72000
# Therefore --> [1.0    0.0     0.0   44.0  72000.0]
#                France Germany Spain Age   Salary
# This is ordered alphabetically         
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Encoding the dependent variable, this will label Yes(s) with ones and No(s) with zeros
le = LabelEncoder()
Y = le.fit_transform(Y)

# Splitting the dataset into training(80% observation) and test(20% observation) set with random factor of 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=1)

# Feature Scaling which optimize the accuracy of the model predictions
# Normalisation is used when normal distribution was involved in the model
# Standardisation is applicable in all model situation hence, this approach is chosen
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:]) # X_test require the same scaler as the one used in X_train

print(f"X:\n{X}\n")
print(f"X_train:\n{X_train}\n")
print(f"X_test:\n{X_test}\n")
print("--------------------------------------")
print(f"Y:\n{Y}\n")
print(f"Y_train:\n{Y_train}\n")
print(f"Y_test:\n{Y_test}\n")


