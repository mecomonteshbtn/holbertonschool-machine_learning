#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  28 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images using multiple kernels

    Arguments:
     - images is a numpy.ndarray with shape (m, h, w, c) containing
        multiple images
        * m is the number of images
        * h is the height in pixels of the images
        * w is the width in pixels of the images
        * c is the number of channels in the image
     - kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the
        kernels for the convolution
        * kh is the height of a kernel
        * kw is the width of a kernel
        * nc is the number of kernels
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

    m, h, w, c = images.shape
    hk, wk, c, cn = kernels.shape
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
    outputs = np.zeros((m, h_output, w_output, cn))

    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)], 'constant',
                    constant_values=0)

    for i in range(h_output):
        for j in range(w_output):
            for nc in range(c):
                x = hk + i * hs
                y = wk + j * ws
                outputs[:, i, j, nc] = np.sum(np.multiply(images[:, i*hs:x,
                                                                 j*ws:y],
                                                          kernels[:, :, :,
                                                                  nc]),
                                              axis=(1, 2, 3))

    return outputs
