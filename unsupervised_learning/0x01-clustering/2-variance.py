#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np


def variance(X, C):
    """
    Function that calculates the total intra-cluster variance for a data set

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the data set
     - C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster

    Returns:
     var, or None on failure
         - var is the total variance
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(X.shape) != 2:
        return None

    try:
        n, d = X.shape
        distance = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        min_distance = np.min(distance, axis=0)
        v = np.sum(min_distance ** 2)

        return v

    except Exception:
        return None
