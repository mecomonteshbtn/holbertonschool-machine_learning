#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  28 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images:

    Arguments:
     - images is a numpy.ndarray with shape (m, h, w) containing
        multiple grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
     - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel

    Returns:
     A numpy.ndarray containing the convolved images
    """

    m, h, w = images.shape
    hk, wk = kernel.shape

    h_output = h - hk + 1
    w_output = w - wk + 1

    ph = (hk - int(hk % 2 == 1)) // 2
    pw = (wk - int(wk % 2 == 1)) // 2
    h_output = h - hk + 2 * ph + int(hk % 2 == 1)
    w_output = w - wk + 2 * pw + int(wk % 2 == 1)

    outputs = np.zeros((m, h_output, w_output))
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant',
                    constant_values=0)

    for i in range(h_output):
        for j in range(w_output):
            x = hk + i
            y = wk + j
            outputs[:, i, j] = np.sum(np.multiply(images[:, i:x, j:y],
                                                  kernel), axis=(1, 2))

    return outputs
