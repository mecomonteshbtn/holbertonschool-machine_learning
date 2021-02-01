#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  28 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on grayscale images

    Arguments:
     - images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
     - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel
     - padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            * ph is the padding for the height of the image
            * pw is the padding for the width of the image
        the image should be padded with 0’s
     - stride is a tuple of (sh, sw)
        * sh is the stride for the height of the image
        * sw is the stride for the width of the image

    Returns:
     A numpy.ndarray containing the convolved images
    """

    m, h, w = images.shape
    hk, wk = kernel.shape
    hs, ws = stride

    if padding == 'same':
        ph = ((h - 1) * hs + hk - h) // 2 + 1
        pw = ((w - 1) * ws + wk - w) // 2 + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    if type(padding) is tuple:
        ph, pw = padding

    h_output = (h - hk + 2 * ph) // hs + 1
    w_output = (w - wk + 2 * pw) // ws + 1
    outputs = np.zeros((m, h_output, w_output))

    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant',
                    constant_values=0)

    for i in range(h_output):
        for j in range(w_output):
            x = hk + i * hs
            y = wk + j * ws
            outputs[:, i, j] = np.sum(np.multiply(images[:, i*hs:x, j*ws:y],
                                                  kernel), axis=(1, 2))

    return outputs
