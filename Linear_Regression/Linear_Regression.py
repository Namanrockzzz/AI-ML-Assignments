
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
X_train = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Linear_Regression/Linear_X_Train.csv")
Y_train = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Linear_Regression/Linear_Y_Train.csv")
X_test = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Linear_Regression/Linear_X_Test.csv")

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

# Generating CSV for output
import csv
with open(r"//home/namanrockzzz/Documents/AI_ML Novice/Linear_Regression/Linear_Y_Test.csv",'w',newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(["y"])
    for value in Y_pred:
        writer.writerow(value)