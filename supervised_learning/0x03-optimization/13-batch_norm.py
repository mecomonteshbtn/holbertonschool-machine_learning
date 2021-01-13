#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output of a NN using
    batch normalization
    Arguments:
     - Z is a numpy.ndarray of shape (m, n) that should be normalized
         * m is the number of data points
         * n is the number of features in Z
     - gamma is a numpy.ndarray of shape (1, n) containing the scales
        used for batch normalization
     - beta is a numpy.ndarray of shape (1, n) containing the offsets
        used for batch normalization
     - epsilon is a small number used to avoid division by zero
    Returns:
    The normalized Z matrix
    """

    mt = Z.mean(0)
    vt = Z.var(0)

    Zt = (Z - mt) / (vt + epsilon) ** (1/2)
    normalized_Z = gamma * Zt + beta

    return normalized_Z
