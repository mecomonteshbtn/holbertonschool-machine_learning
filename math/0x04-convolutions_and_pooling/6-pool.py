#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  28 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Funtion that performs pooling on images

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
     - stride is a tuple of (sh, sw)
        * sh is the stride for the height of the image
        * sw is the stride for the width of the image
     - mode indicates the type of pooling
        * max indicates max pooling
        * avg indicates average pooling

    Returns:
     A numpy.ndarray containing the pooled images
    """

    m, h, w, c = images.shape
    hk, wk = kernel_shape
    hs, ws = stride

    h_output = (h - hk) // hs + 1
    w_output = (w - wk) // ws + 1
    pooled = np.zeros((m, h_output, w_output, c))

    for i in range(h_output):
        for j in range(w_output):
            x = hk + i * hs
            y = wk + j * ws
            if mode == 'max':
                data = np.max(images[:, i*hs:x, j*ws:y], axis=(1, 2))
            if mode == 'avg':
                data = np.mean(images[:, i*hs:x, j*ws:y], axis=(1, 2))
            pooled[:, i, j] = data

    return pooled
