# SVM

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Classification(SVM,Kernel_SVM,Naive_Bayes)/pulsar_stars.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Saving test and training sets into csvs
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification(SVM,Kernel_SVM,Naive_Bayes)/SVM_Classification/X_train.csv", X_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification(SVM,Kernel_SVM,Naive_Bayes)/SVM_Classification/X_test.csv", X_test, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification(SVM,Kernel_SVM,Naive_Bayes)/SVM_Classification/Y_train.csv", Y_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification(SVM,Kernel_SVM,Naive_Bayes)/SVM_Classification/Y_test.csv", Y_test, delimiter = ",")

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting SVM classifier to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Saving Y_pred into csv
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification(SVM,Kernel_SVM,Naive_Bayes)/SVM_Classification/Y_pred.csv", Y_pred, delimiter = ",")

# Making the confusion matrix
# it tells how many predictions were correct and how many were incorrect
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

# By comparing Y_test and Y_pred
# We can see that there are only 119 incorrect values out of 5966 values