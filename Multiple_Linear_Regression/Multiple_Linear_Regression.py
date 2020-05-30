
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Multiple_Linear_Regression/Train.csv")
X_train = dataset.iloc[:, :-1].values
Y_train = dataset.iloc[:,-1].values
X_test =  pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Multiple_Linear_Regression/Test.csv")

# importing linear regression library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

# Generating CSV for output
np.savetxt(r"/home/namanrockzzz/Documents/AI_ML Novice/Multiple_Linear_Regression/Predicted_target_values.csv", Y_pred, delimiter=",")

# Building optimal model using Backward Elimination
import statsmodels.api as sm
X = X_train
Y = Y_train
X = np.append(arr = np.ones((1600,1)).astype(int), values = X_train ,axis = 1)
X_opt = np.array(X[:,[0,1,2,3,4,5]],dtype = float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
# Since P-value for each variable is less than 5% therfore this is the most optimal model