#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def correlation(C):
    """
    Function that calculates a correlation matrix

    Arguments:
     - C is a numpy.ndarray of shape (d, d)
        containing a covariance matrix
        * d is the number of dimensions

    Returns
     A numpy.ndarray of shape (d, d) containing the correlation matrix
    """

    if type(C) != np.ndarray:
        raise TypeError('C must be a numpy.ndarray')

    if len(C.shape) != 2:
        raise ValueError('C must be a 2D square matrix')

    d1 = C.shape[0]
    d2 = C.shape[1]

    if d1 != d2:
        raise ValueError('C must be a 2D square matrix')

    variance = np.diag(1 / np.sqrt(np.diag(C)))
    corr = np.matmul(np.matmul(variance, C), variance)

    return corr
