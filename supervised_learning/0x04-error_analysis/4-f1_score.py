#!/usr/bin/env python3
"""
Function f1_score
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Function that calculates the F1 score of a confusion matrix
    Arguments:
    - confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes is the number of classes
    Returns:
    A numpy.ndarray of shape (classes,) containing the F1 score of each class
    """
    _sensitivity = sensitivity(confusion)
    _precision = precision(confusion)
    F1_score = 2 * ((_precision * _sensitivity) / (_precision + _sensitivity))

    return F1_score
