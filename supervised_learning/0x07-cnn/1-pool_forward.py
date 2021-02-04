#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs forward propagation over a pooling layer of a NN

    Arguments:
     - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
     - kernel_shape is a tuple of (kh, kw) containing the size of
        the kernel for the pooling
        * kh is the kernel height
        * kw is the kernel width
     - stride is a tuple of (sh, sw) containing the strides for
        the convolution
        * sh is the stride for the height
        * sw is the stride for the width
     - mode is a string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively

    Returns:
     The output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_output = (h_prev - kh) // sh + 1
    w_output = (w_prev - kw) // sw + 1
    outputs = np.zeros((m, h_output, w_output, c_prev))

    for h in range(h_output):
        for w in range(w_output):
            x = kh + h * sh
            y = kw + w * sw
            if mode == 'max':
                data = np.max(A_prev[:, h*sh:x, w*sw:y], axis=(1, 2))
            if mode == 'avg':
                data = np.mean(A_prev[:, h*sh:x, w*sw:y], axis=(1, 2))
            outputs[:, h, w] = data

    return outputs
