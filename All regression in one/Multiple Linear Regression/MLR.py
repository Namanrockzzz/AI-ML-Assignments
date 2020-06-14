# Multiple Linear Regression

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
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Multiple Linear Regression/X_train.csv", X_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Multiple Linear Regression/X_test.csv", X_test, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Multiple Linear Regression/Y_train.csv", Y_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Multiple Linear Regression/Y_test.csv", Y_test, delimiter = ",")

# importing linear regression library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting values
Y_pred = regressor.predict(X_test)

# Saving predicted values into a CSV
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Multiple Linear Regression/Y_pred.csv", Y_pred, delimiter = ",")

# Building optimal model using Backward Elimination
import statsmodels.api as sm
# this library does not take into account the constant of our
# multiple regression model so we will have to add a column 
# in our matrix of features containing only 1 which will make
# this library to consider b0*x0 where x0 is always 1
X = np.append(arr = np.ones((414,1)).astype(int), values = X ,axis = 1)
X_opt = np.array(X[:,[0,1,2,3,4,5,6]],dtype = float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# OLS stands for ordinary least square
regressor_OLS.summary()
# By this we check the P values for each and every independent variable
# And delete the one with highest p-value
X_opt = np.array(X[:,[0,1,2,3,4,5]],dtype = float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
# now all the p values are less than 0.05 so our multiple linear model is now perfect

