#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 7:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def definiteness(matrix):
    """
    Function that calculates the definiteness of a matrix

    Arguments:
     - matrix is a numpy.ndarray of shape (n, n) whose definiteness
        should be calculated

    Return:
    If the matrix is positive definite, positive semi-definite,
    negative semi-definite, negative definite of indefinite
     the string:
     - Positive definite
     - Positive semi-definite
     - Negative semi-definite
     - Negative definite
     - Indefinite
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) == 1:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.array_equal(matrix.T, matrix):
        return None

    w, v = np.linalg.eig(matrix)

    if np.all(w > 0):
        return 'Positive definite'
    if np.all(w >= 0):
        return 'Positive semi-definite'
    if np.all(w < 0):
        return 'Negative definite'
    if np.all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
