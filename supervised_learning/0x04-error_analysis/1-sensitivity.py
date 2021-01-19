#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 08:42:40 2021

@author: Robinson Montes
"""
import numpy as np


def sensitivity(confusion):
    """
    Function that calculates the sensitivity for each class
    in a confusion matrix
    Arguments:
     - confusion: is a confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent the correct labels and column
                  indices represent the predicted labels
        * classes is the number of classes
    Returns:
    A numpy.ndarray of shape (classes,) containing the sensitivity
    of each class
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    SENSITIVITY = TP / (TP + FN)

    return SENSITIVITY
