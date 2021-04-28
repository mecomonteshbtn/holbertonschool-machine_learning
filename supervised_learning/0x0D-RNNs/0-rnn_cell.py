#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  28 7:26:32 2021

@author: Robinson Montes
"""
import numpy as np


class RNNCell():
    """
    Class RNNCell that represents a cell of a simple RNN:
    """
    def __init__(self, i, h, o):
        """
        Constructor

        Argumetns:
         - i is the dimensionality of the data
         - h is the dimensionality of the hidden state
         - o is the dimensionality of the outputs

         Public instance attributes
          - Wh, Wy, bh, by that represent the weights and biases of the cell
            * Wh and bh are for the concatenated hidden state and input data
            * Wy and by are for the output
        """

        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
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

        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, y
