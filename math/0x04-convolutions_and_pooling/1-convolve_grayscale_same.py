#!/usr/bin/env python3
"""
Same Convolution
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

    m = images.shape[0]
    himage = images.shape[1]
    wimage = images.shape[2]
    hkernel = kernel.shape[0]
    wkernel = kernel.shape[1]

    if not hkernel % 2:
        ph = int(hkernel / 2)
        hfinal = himage - hkernel + (2 * ph)
    else:
        ph = int((hkernel - 1) / 2)
        hfinal = himage - hkernel + 1 + (2 * ph)

    if not wkernel % 2:
        pw = int(wkernel / 2)
        wfinal = wimage - wkernel + (2 * pw)
    else:
        pw = int((wkernel - 1) / 2)
        wfinal = wimage - wkernel + 1 + (2 * pw)

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
