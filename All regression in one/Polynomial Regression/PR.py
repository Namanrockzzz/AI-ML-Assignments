# Polynomial Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/datasets_88705_204267_Real estate.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:,-1:].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Saving test and training sets into CSVs
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Polynomial Regression/X_train.csv", X_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Polynomial Regression/X_test.csv", X_test, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Polynomial Regression/Y_train.csv", Y_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Polynomial Regression/Y_test.csv", Y_test, delimiter = ",")

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# poly_reg = PolynomialFeatures(degree = 4)
# poly_reg = PolynomialFeatures(degree = 3)
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y_train)

# Converting X_test into poly version
X_poly_test = poly_reg.transform(X_test)

# Predicting a new result with polynomial regression
Y_pred = lin_reg.predict(X_poly_test)

# Saving predicted values into a CSV
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Polynomial Regression/Y_pred.csv", Y_pred, delimiter = ",")