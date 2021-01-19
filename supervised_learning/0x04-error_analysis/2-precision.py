#!/usr/bin/env python3
"""
Precision function
"""

import numpy as np


def precision(confusion):
    """
    Function that calculates the precision for each class in a confusion matrix
    Arguments:
     - confusion is a confusion numpy.ndarray of shape (classes, classes) where
                 row indices represent the correct labels and column indices
                 represent the predicted labels.
                 classes is the number of classes
    Returns:
    A numpy.ndarray of shape (classes,) containing the precision of each class
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    PRECISION = TP / (TP + FP)

    return PRECISION
