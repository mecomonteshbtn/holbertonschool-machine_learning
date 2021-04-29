#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  28 7:26:32 2021

@author: Robinson Montes
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Function that performs forward propagation for a simple RNN

    Arguments:
     - rnn_cell is an instance of RNNCell that will be used
        for the forward propagation
     - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        * t is the maximum number of time steps
        * m is the batch size
        * i is the dimensionality of the data
     - h_0 is the initial hidden state, given as
        a numpy.ndarray of shape (m, h)
        * h is the dimensionality of the hidden state

    Returns:
     H, Y
        - H is a numpy.ndarray containing all of the hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """

    t, m, i = X.shape
    _, h = h_0.shape

    Y = []
    H = np.zeros((t + 1, m, h))
    H[0, :, :] = h_0
    for step in range(t):
        h, y = rnn_cell.forward(H[step], X[step])
        H[step + 1, :, :] = h
        Y.append(y)
    Y = np.asarray(Y)

    return H, Y
