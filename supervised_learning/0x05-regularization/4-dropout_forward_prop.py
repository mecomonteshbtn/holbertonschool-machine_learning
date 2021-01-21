#!/usr/bin/env python3
"""
Forward Propagation with Dropout
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that conducts forward propagation using Dropout
    Arguments:
    - X is a numpy.ndarray of shape (nx, m) containing the input data
        for the network
        * nx is the number of input features
        * m is the number of data points
     - weights is a dictionary of the weights and biases of the neural network
     - L the number of layers in the network
     - keep_prob is the probability that a node will be kept
    Returns:
     A dictionary containing the outputs of each layer and the dropout mask
     used on each layer
    """

    cache = {}
    cache['A0'] = X

    for layer in range(L):
        Al_key = 'A' + str(layer)
        w_key = 'W' + str(layer + 1)
        b_key = 'b' + str(layer + 1)
        dl_key = 'D' + str(layer + 1)
        Al1_key = 'A' + str(layer + 1)

        Al = cache[Al_key]
        Wl = weights[w_key]
        bl = weights[b_key]
        Zl = np.matmul(Wl, Al) + bl
        if layer != L - 1:
            a = np.sinh(Zl) / np.cosh(Zl)
            dl = np.random.binomial(1, keep_prob, (a.shape[0], a.shape[1]))
            cache[dl_key] = dl
            a = np.multiply(a, dl)
            cache[Al1_key] = a / keep_prob
        else:
            t = np.exp(Zl)
            a = np.exp(Zl) / np.sum(t, axis=0, keepdims=True)
            cache[Al1_key] = a

    return cache
