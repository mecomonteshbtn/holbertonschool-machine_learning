#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.show()

Kmean = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
               n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
               random_state=None, tol=0.0001, verbose=0)
Kmean.fit(X)
Kmean.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()

Kmean.labels_
sample_test = np.array([-3.0, -3.0])
second_test = sample_test.reshape(1, -1)
result = Kmean.predict(second_test)
print(result)
