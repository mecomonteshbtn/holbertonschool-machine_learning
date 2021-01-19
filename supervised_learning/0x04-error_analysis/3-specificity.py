#!/usr/bin/env python3
"""
Function specificity
"""


import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity
    for each class in a confusion matrix
    Arguments:
     - confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes is the number of classes
    Returns:
    A numpy.ndarray of shape (classes,) containing
    the specificity of each class
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    SPECIFICITY = TN / (FP + TN)

    return SPECIFICITY
