#!/usr/bin/env python3
"""
Gradient Descent with Dropout
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that updates the weights of a NN with Dropout regularization
    using gradient descent
    Arguments:
     - Y is a one-hot numpy.ndarray of shape (classes, m) that contains
         the correct labels for the data
         * classes is the number of classes
         * m is the number of data points
     - weights is a dictionary of the weights and biases of the NN
     - cache is a dictionary of the outputs and dropout masks of
        each layer of the NN
     - alpha is the learning rate
     - keep_prob is the probability that a node will be kept
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
        D_key = 'D' + str(layer)

        Al = cache[Al_key]
        gld = 1 - np.power(Al, 2)
        if layer == L:
            dZl = dAl
        else:
            dZl = dAl * gld
            dZl *= cache[D_key] / keep_prob

        Wl = weights[w_key]
        Al1 = cache[Al1_key]
        dWl = (1 / m) * np.matmul(dZl, Al1.T)
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
        dAl = np.matmul(Wl.T, dZl)
        weights[w_key] = weights[w_key] - alpha * dWl
        weights[b_key] = weights[b_key] - alpha * dbl
