#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 4 8:43:40 2021

@author: Robinson Montes
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix:

    Arguments:
    - Y is a numpy.ndarray with shape (m,) containing numeric class labels
    - m is the number of examples
    classes is the maximum number of classes found in Y

    Returns:
    a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if ((type(Y) is not np.ndarray) or (len(Y) == 0) or
       (type(classes) is not int) or (classes <= np.amax(Y))):
        return None
    return np.eye(classes)[Y].T
