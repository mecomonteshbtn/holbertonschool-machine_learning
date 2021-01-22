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

    weights_copy = weights.copy()
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        w = "W" + str(i)
        b = "b" + str(i)
        dw = (1 / len(Y[0])) * np.matmul(dz, A.T) + (
            lambtha * weights[w]) / len(Y[0])
        db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
        weights[w] = weights[w] - alpha * dw
        weights[b] = weights[b] - alpha * db
        dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (1 - A * A)
