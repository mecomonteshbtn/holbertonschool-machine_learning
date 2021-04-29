#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  28 7:26:32 2021

@author: Robinson Montes
"""
import numpy as np


# Function
def sigmoid(x):
    """
    Sigmoid function
    """
    sigmoid = 1 / (1 + np.exp(-x))

    return sigmoid


class GRUCell():
    """
    Class GRUCell that represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        class constructor

        Argumetns:
         - i is the dimensionality of the data
         - h is the dimensionality of the hidden state
         - o is the dimensionality of the outputs

        Public instance attributes
         Wz, Wr, Wh, Wy, bz, br, bh, by that represent the weights and biases
         of the cell
         - Wzand bz are for the update gate
         - Wrand br are for the reset gate
         - Whand bh are for the intermediate hidden state
         - Wyand by are for the output
        """

        # Weights
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        # Bias
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    # Function
    def softmax(self, x):
        """
        Function to compute softmax values for each sets of scores in x
        """

        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)

        return e_x / np.sum(e_x, axis=1, keepdims=True)

    # public instance method
    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation
        for one time step

        Arguments:
         - x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            * m is the batche size for the data
         - h_prev is a numpy.ndarray of shape (m, h) containing
            the previous hidden state

        Returns:
         h_next, y
            - h_next is the next hidden state
            - y is the output of the cell
        """

        matrix = np.concatenate((h_prev, x_t), axis=1)
        z_t = sigmoid(np.matmul(matrix, self.Wz) + self.bz)
        r_t = sigmoid(np.matmul(matrix, self.Wr) + self.br)

        matrix2 = np.concatenate((r_t * h_prev, x_t), axis=1)
        prime_h = np.tanh(np.matmul(matrix2, self.Wh) + self.bh)
        # s_t = (1 - z) * s_t-1 + z * h
        h_next = (1 - z_t) * h_prev + z_t * prime_h

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
