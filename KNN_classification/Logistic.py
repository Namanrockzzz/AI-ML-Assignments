# Logistic Regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Test and Training set
X_train = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/KNN_classification/Diabetes_XTrain.csv")
Y_train = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/KNN_classification/Diabetes_YTrain.csv")
X_test = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/KNN_classification/Diabetes_Xtest.csv")

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred2 = classifier.predict(X_train)

# Making the confusion matrix
# it tells how many predictions were correct and how many were incorrect
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train,Y_pred2)
# Here we can see that out of 576 values 453 are correct and 123 are incorrect

# Saving predicted values into a CSV
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/KNN_classification/Y_pred_Logistic_regression.csv", Y_pred, delimiter = ",")