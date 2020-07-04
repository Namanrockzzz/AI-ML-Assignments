# Random Forest classification

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv(r"/home/namanrockzzz/Documents/AI_ML Novice/Classification( Decision_tree, Random_forest)/datasets_4458_8204_winequality-red.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Categorising quality of wine to good/1 or bad/1
for i in range(len(Y)):
    if Y[i]>=7:
        Y[i]=1
    else:
        Y[i]=0

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Saving test and training sets into csvs
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification( Decision_tree, Random_forest)/Random Forest/X_train.csv", X_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification( Decision_tree, Random_forest)/Random Forest/X_test.csv", X_test, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification( Decision_tree, Random_forest)/Random Forest/Y_train.csv", Y_train, delimiter = ",")
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification( Decision_tree, Random_forest)/Random Forest/Y_test.csv", Y_test, delimiter = ",")


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting random forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = "entropy", random_state = 0)
classifier.fit(X_train, Y_train)


# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Saving Y_pred into csv
np.savetxt("/home/namanrockzzz/Documents/AI_ML Novice/Classification( Decision_tree, Random_forest)/Random Forest/Y_pred.csv", Y_pred, delimiter = ",")

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

# by comparing Y_test and Y_pred
# We can see that only 42 values out of 533 values are predicted incorrect i.e AR = 92.12%