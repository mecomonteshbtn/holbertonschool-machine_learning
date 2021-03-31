#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Function that performs K-means on a dataset

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the dataset
        * n is the number of data points
        * d is the number of dimensions for each data point
     - k is a positive integer containing the number of clusters
     - iterations is a positive integer containing the maximum number of
        iterations that should be performed

    Returns:
     C, clss, or None, None on failure
         - C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
         - clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if type(k) != int or k <= 0:
        return None, None

    if type(iterations) != int or iterations <= 0:
        return None, None

    n, d = X.shape

    minimum = np.amin(X, axis=0)
    maximum = np.amax(X, axis=0)

    # C = np.random.uniform(minimum, maximum, (k, d))
    C = initialize(X, k)
    clss = None
    for i in range(iterations):
        C_cpy = np.copy(C)
        distance = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distance, axis=-1)
        # move the centroids
        for j in range(k):
            index = np.argwhere(clss == j)
            if not len(index):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[index], axis=0)

        if (C_cpy == C).all():
            return C, clss

    distance = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(distance, axis=-1)

    return C, clss


def initialize(X, k):
    """
    Function that initializes cluster centroids for K-means

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the dataset
         that will be used for K-means clustering
        * n is the number of data points
        * d is the number of dimensions for each data point
     - k is a positive integer containing the number of clusters

    Returns:
     A numpy.ndarray of shape (k, d) containing the initialized centroids
     for each cluster, or None on failure
    """

    n, d = X.shape

    minimum = np.amin(X, axis=0)
    maximum = np.amax(X, axis=0)

    values = np.random.uniform(minimum, maximum, (k, d))

    return values
