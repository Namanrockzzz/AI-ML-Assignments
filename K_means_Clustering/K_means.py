#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:30:59 2020

@author: namanrockzzz
"""


# Importing librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing mall dataset with pandas
dataset = pd.read_csv("datasets_721951_1255613_Country-data.csv")
# Taking out required data from dataset
X = dataset.iloc[:,1:].values
# Saving the list of countries
countries = dataset.iloc[:,[0]].values

# Using the elow method to find the optimal numbers of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(i, init = 'k-means++',n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11), WCSS)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("wcss value")
plt.show() # By looking at Dendogram we found that k=4 is optimal solution to the number of clusters

# Applying K-means to the mall dataset(k=3)
kmeans = KMeans(4, init = 'k-means++',n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Getting centroid of each cluster
centroids = kmeans.cluster_centers_ 
# By analyzing centroids we can say that cluster 0 contains the countries which can be termed as poorest based on socio-economic and health factors

# Making a list of the countries that are in the direst need of aid
poor_countries = []
for i in range(167):
    if y_kmeans[i] == 0:
        poor_countries.append(countries[i][0])
poor_countries = np.array(poor_countries)
poor_countries.reshape((108,1))

# Saving this list in a CSV
np.savetxt("Countries_to_focus_more_on.csv", poor_countries, delimiter = ",",fmt = '%s')

# Conclusion out of 167 countries 108 countries are in the direst need of aid