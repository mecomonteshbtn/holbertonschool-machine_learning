#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  28 7:26:32 2021

@author: Robinson Montes
"""
import numpy as np


# Function
def softmax(x):
    """
    Function to compute softmax values for each sets of scores in x
    """

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class BidirectionalCell():
    """
    Class BidirectionalCell that represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Argumetns:
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs

        Public instance attributes:
         Whf, Whb, Wy, bhf, bhb, by that represent
         the weights and biases of the cell
          - Whf and bhfare for the hidden states in the forward direction
          - Whb and bhbare for the hidden states in the backward direction
          - Wy and byare for the outputs
        """

        # Weights
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        # Bias
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    # public instance method
    def forward(self, h_prev, x_t):
        """
        Public instance method that calculates the hidden state in
        the forward direction for one time step

        Arguments:
         - x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            * m is the batch size for the data
         - h_prev is a numpy.ndarray of shape (m, h) containing
            the previous hidden state

        Returns:
         h_next, the next hidden state
        """

        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next

    # Public instance method
    def backward(self, h_next, x_t):
        """
        Public instance method that calculates the hidden state
        in the backward direction for one time step

        Arguments:
         - x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            * m is the batch size for the data
         - h_next is a numpy.ndarray of shape (m, h) containing
            the next hidden state

        Returns:
         h_pev, the previous hidden state
        """

        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_prev

    # Public instance method
    def output(self, H):
        """
        Public instance method that calculates all outputs for the RNN:

        Arguments:
         - H is a numpy.ndarray of shape (t, m, 2 * h) that contains
            the concatenated hidden states from both directions,
            excluding their initialized states
            * t is the number of time steps
            * m is the batch size for the data
            * h is the dimensionality of the hidden states

        Returns:
         Y, the outputs
        """

        t = H.shape[0]
        Y = []

        for time in range(t):
            y = np.matmul(H[time], self.Wy) + self.by
            y = softmax(y)
            Y.append(y)

        return np.array(Y)
