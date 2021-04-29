#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  28 7:26:32 2021

@author: Robinson Montes
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function that performs forward propagation for a deep RNN

    Arguments:
     - rnn_cells is a list of RNNCell instances of length l
        that will be used for the forward propagation
        * l is the number of layers
     - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        * t is the maximum number of time times
        * m is the batch size
        * i is the dimensionality of the data
     - h_0 is the initial hidden state, given as a numpy.ndarray
        of shape (l, m, h)
        * h is the dimensionality of the hidden state

    Returns:
     H, Y
        - H is a numpy.ndarray containing all of the hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """

    t_layers = len(rnn_cells)
    t, m, i = X.shape
    _, _, h_ = h_0.shape

    H = np.zeros((t + 1, t_layers, m, h_))
    H[0] = h_0

    for time in range(t):
        for layer in range(t_layers):
            if layer == 0:
                h, y = rnn_cells[layer].forward(H[time, layer], X[time])
            else:
                h, y = rnn_cells[layer].forward(H[time, layer], h)

            H[time + 1, layer, ...] = h

            if layer == t_layers - 1:
                if time == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

    Y = Y.reshape(t, m, Y.shape[-1])

    return H, Y
