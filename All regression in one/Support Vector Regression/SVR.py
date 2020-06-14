# Support Vector Regression

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
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Support Vector Regression/X_train.csv", X_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Support Vector Regression/X_test.csv", X_test, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Support Vector Regression/Y_train.csv", Y_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Support Vector Regression/Y_test.csv", Y_test, delimiter = ",")

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Y_train = sc_Y.fit_transform(Y_train)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,Y_train)

# Predicting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(X_test))

# Saving predicted values into a CSV
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/All regression in one/Support Vector Regression/Y_pred.csv", Y_pred, delimiter = ",")
