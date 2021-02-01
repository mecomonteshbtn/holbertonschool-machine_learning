#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  28 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Fucntion that performs a convolution on grayscale images
    with custom padding

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
     - padding is a tuple of (ph, pw)
        * ph is the padding for the height of the image
        * pw is the padding for the width of the image
        the image should be padded with 0’s

    Returns:
     A numpy.ndarray containing the convolved images
    """

    m, h, w = images.shape
    hk, wk = kernel.shape
    ph, pw = padding

    h_output = h + 2 * ph - hk + 1
    w_output = w + 2 * pw - wk + 1

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
