#!/usr/bin/env python3
"""
Multiple kernels
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

    m = images.shape[0]
    himage = images.shape[1]
    wimage = images.shape[2]
    ncimage = images.shape[3]
    hkernel = kernels.shape[0]
    wkernel = kernels.shape[1]
    nckernel = kernels.shape[3]
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
    convoluted = np.zeros((m, hfinal, wfinal, nckernel))

    mImage = np.arange(0, m)
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)], 'constant',
                    constant_values=0)

    for i in range(hfinal):
        for j in range(wfinal):
            for nc in range(nckernel):
                data = np.sum(np.multiply(images[mImage,
                                                 i*sh:hkernel+(i*sh),
                                                 j*sw:wkernel+(j*sw)],
                              kernels[:, :, :, nc]), axis=(1, 2, 3))
                convoluted[mImage, i, j, nc] = data

    return convoluted
