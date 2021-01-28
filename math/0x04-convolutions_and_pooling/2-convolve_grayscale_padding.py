#!/usr/bin/env python3
"""
Convolution with Padding
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

    m = images.shape[0]
    himage = images.shape[1]
    wimage = images.shape[2]
    hkernel = kernel.shape[0]
    wkernel = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]

    hfinal = himage + (2 * ph) - hkernel + 1
    wfinal = wimage + (2 * pw) - wkernel + 1

    convoluted = np.zeros((m, hfinal, wfinal))

    mImage = np.arange(0, m)
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant',
                    constant_values=0)

    for i in range(hfinal):
        for j in range(wfinal):
            data = np.sum(np.multiply(images[mImage, i:hkernel+i, j:wkernel+j],
                          kernel), axis=(1, 2))
            convoluted[mImage, i, j] = data

    return convoluted
