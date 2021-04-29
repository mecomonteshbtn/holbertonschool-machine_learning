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


class LSTMCell():
    """
    Class LSTMCell that represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
        class constructor

        Argumetns:
         - i is the dimensionality of the data
         - h is the dimensionality of the hidden state
         - o is the dimensionality of the outputs

        Public instance attributes
         Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by that represent
         the weights and biases of the cell
         - Wf and bf are for the forget gate
         - Wu and bu are for the update gate
         - Wc and bc are for the intermediate cell state
         - Wo and bo are for the output gate
         - Wy and by are for the outputs
        """

        # Weights
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        # Bias
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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
    def forward(self, h_prev, c_prev, x_t):
        """
        Public instance method that performs forward propagation
        for one time step

        Arguments:
         - x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
            * m is the batche size for the data
         - h_prev is a numpy.ndarray of shape (m, h) containing
            the previous hidden state
         - c_prev is a numpy.ndarray of shape (m, h) containing
            the previous cell state

        Returns:
         h_next, c_next, y
            - h_next is the next hidden state
            - c_next is the next cell state
            - y is the output of the cell
        """

        matrix = np.concatenate((h_prev, x_t), axis=1)
        u_t = sigmoid(np.matmul(matrix, self.Wu) + self.bu)
        f_t = sigmoid(np.matmul(matrix, self.Wf) + self.bf)
        o_t = sigmoid(np.matmul(matrix, self.Wo) + self.bo)

        prime_c = np.tanh(np.matmul(matrix, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * prime_c
        h_next = o_t * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
