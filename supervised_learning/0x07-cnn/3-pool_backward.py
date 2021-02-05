#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function  that performs back propagation over a pooling layer of a NN

    Arguments:
     - dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the output of the pooling layer
        * m is the number of examples
        * h_new is the height of the output
        * w_new is the width of the output
        * c_new is the number of channels in the output
     - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
     - kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
        * kh is the kernel height
        * kw is the kernel width
     - stride is a tuple of (sh, sw) containing the strides for
        the convolution
        * sh is the stride for the height
        * sw is the stride for the width
     - mode is a string containing either max or avg, indicating whether
        to perform maximum or average pooling, respectively

    Returns:
     The partial derivatives with respect to the previous layer (dA_prev)
    """

    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dx = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for cn in range(c_new):
                    x = kh + h * sh
                    y = kw + w * sw
                    if mode == 'max':
                        aux = A_prev[i, h*sh:x, w*sw:y, cn]
                        mask = np.where(aux == np.max(aux), 1, 0)
                        dx[i, h*sh:x, w*sw:y, cn] += dA[i, h, w, cn] * mask
                    else:
                        avg = dA[i, h, w, cn] / (kw * kh)
                        dx[i,
                           h*sh:x, w*sw:y, cn] += np.ones(kernel_shape) * avg

    return dx
