#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:23:08 2021

@author: Robinson Montes
"""
import numpy as np


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

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None

    n, d = X.shape

    minimum = X.min(axis=0)
    maximum = X.max(axis=0)

    values = np.random.uniform(minimum, maximum, (k, d))

    return values
