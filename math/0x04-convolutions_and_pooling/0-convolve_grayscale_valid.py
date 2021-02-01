#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  28 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images:

    Arguments:
     - images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
         * m is the number of images
         * h is the height in pixels of the images
         * w is the width in pixels of the images
     - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel

    Returns:
    A numpy.ndarray containing the convolved images
    """

    m, h, w = images.shape
    hk, wk = kernel.shape

    h_output = h - hk + 1
    w_output = w - wk + 1

    outputs = np.zeros((m, h_output, w_output))

    for i in range(h_output):
        for j in range(w_output):
            x = hk + i
            y = wk + j
            outputs[:, i, j] = np.sum(np.multiply(images[:, i:x, j:y],
                                                  kernel), axis=(1, 2))

    return outputs
