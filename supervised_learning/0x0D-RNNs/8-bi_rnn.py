#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  28 7:26:32 2021

@author: Robinson Montes
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Function that performs forward propagation for a bidirectional RNN

    Arguments:
     - bi_cells is an instance of BidirectinalCell
        that will be used for the forward propagation
     - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        * t is the maximum number of time steps
        * m is the batch size
        * i is the dimensionality of the data
     - h_0 is the initial hidden state in the forward direction,
        given as a numpy.ndarray of shape (m, h)
        * h is the dimensionality of the hidden state
     - h_t is the initial hidden state in the backward direction,
        given as a numpy.ndarray of shape (m, h)

    Returns:
     H, Y
        - H is a numpy.ndarray containing all of the concatenated hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """

    t, m, i = X.shape
    _, h = h_0.shape
    H_f = np.zeros((t + 1, m, h))
    H_b = np.zeros((t + 1, m, h))
    H_f[0] = h_0
    H_b[t] = h_t

    for t_i in range(t):
        H_f[t_i + 1] = bi_cell.forward(H_f[t_i], X[t_i])

    for t_j in range(t - 1, -1, -1):
        H_b[t_j] = bi_cell.backward(H_b[t_j + 1], X[t_j])

    H = np.concatenate((H_f[1:], H_b[0:t]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
