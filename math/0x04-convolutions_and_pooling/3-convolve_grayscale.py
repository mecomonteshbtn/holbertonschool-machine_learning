#!/usr/bin/env python3
"""
Strided Convolution
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

    m = images.shape[0]
    himage = images.shape[1]
    wimage = images.shape[2]
    hkernel = kernel.shape[0]
    wkernel = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = int(((himage - 1) * sh + hkernel - himage) / 2) + 1
        pw = int(((wimage - 1) * sw + wkernel - wimage) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]

    hfinal = int(((himage - hkernel + (2 * ph)) / sh) + 1)
    wfinal = int(((wimage - wkernel + (2 * pw)) / sw) + 1)
    convoluted = np.zeros((m, hfinal, wfinal))

    mImage = np.arange(0, m)
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant',
                    constant_values=0)

    for i in range(hfinal):
        for j in range(wfinal):
            data = np.sum(np.multiply(images[mImage,
                                             i*sh:hkernel+(i*sh),
                                             j*sw:wkernel+(j*sw)],
                          kernel), axis=(1, 2))
            convoluted[mImage, i, j] = data

    return convoluted
