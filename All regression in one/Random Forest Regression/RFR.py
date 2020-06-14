# Random Forest Regression

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
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Random Forest Regression/X_train.csv", X_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Random Forest Regression/X_test.csv", X_test, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Random Forest Regression/Y_train.csv", Y_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Random Forest Regression/Y_test.csv", Y_test, delimiter = ",")

# Fitting the Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train,Y_train)

# Predicting a new result with the regression
Y_pred = regressor.predict(X_test)

# Saving predicted values into a CSV
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Random Forest Regression/Y_pred.csv", Y_pred, delimiter = ",")
