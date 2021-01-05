#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 4 8:43:40 2021

@author: Robinson Montes
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels:

    Arguments:
    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
        - classes is the maximum number of classes
        - m is the number of examples

    Returns:
    a numpy.ndarray with shape (m, ) containing the numeric labels for each
    example, or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
