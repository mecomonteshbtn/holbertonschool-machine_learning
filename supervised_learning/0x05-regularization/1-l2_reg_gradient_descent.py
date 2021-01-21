#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a NN using gradient descent
    with L2 regularization
    Arguments:
     - Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
       correct labels for the data
        * classes is the number of classes
        * m is the number of data points
     - weights is a dictionary of the weights and biases of the NN
     - cache is a dictionary of the outputs of each layer of the NN
     - alpha is the learning rate
     - lambtha is the L2 regularization parameter
     - L is the number of layers of the network
     """

    m = Y.shape[1]
    Al = cache['A' + str(L)]
    dAl = Al - Y

    for layer in reversed(range(1, L + 1)):
        w_key = 'W' + str(layer)
        b_key = 'b' + str(layer)
        Al_key = 'A' + str(layer)
        Al1_key = 'A' + str(layer - 1)

        Al = cache[Al_key]
        gld = 1 - np.square(Al)
        if layer == L:
            dZl = dAl
        else:
            dZl = dAl * gld

        Wl = weights[w_key]
        Al1 = cache[Al1_key]
        dWl = (np.matmul(dZl, Al1.T) + lambtha * Wl / 2 ) / m
        dbl = np.sum(dZl, axis=1, keepdims=True) / m
        dAl = np.matmul(Wl.T, dZl)
        weights[w_key] = weights[w_key] - alpha * dWl
        weights[b_key] = weights[b_key] - alpha * dbl
