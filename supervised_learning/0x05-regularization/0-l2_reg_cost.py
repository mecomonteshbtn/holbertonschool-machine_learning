#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates the cost of a NN with L2 regularization
    Arguments:
     - cost is the cost of the network without L2 regularization
     - lambtha is the regularization parameter
     - weights is a dictionary of the weights and biases (numpy.ndarrays) of
        the NN
     - L is the number of layers in the NN
     - m is the number of data points used
    Returns:
     The cost of the network accounting for L2 regularization
    """

    sum = 0
    for layer in range(1, L + 1):
        key = 'W' + str(layer)
        sum += np.linalg.norm(weights[key])
    L2_cost = cost + lambtha * sum / (2 * m)

    return L2_cost
