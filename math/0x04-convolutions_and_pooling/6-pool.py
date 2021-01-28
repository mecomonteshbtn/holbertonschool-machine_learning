#!/usr/bin/env python3
"""
Pooling
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

    m = images.shape[0]
    himage = images.shape[1]
    wimage = images.shape[2]
    ncimage = images.shape[3]
    hkernel = kernel_shape[0]
    wkernel = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    hfinal = int(((himage - hkernel) / sh) + 1)
    wfinal = int(((wimage - wkernel) / sw) + 1)
    pooled = np.zeros((m, hfinal, wfinal, ncimage))
    mImage = np.arange(0, m)

    for i in range(hfinal):
        for j in range(wfinal):
            if mode == 'max':
                data = np.max(images[mImage,
                                     i*sh:hkernel+(i*sh),
                                     j*sw:wkernel+(j*sw)],
                              axis=(1, 2))
            if mode == 'avg':
                data = np.mean(images[mImage,
                                      i*sh:hkernel+(i*sh),
                                      j*sw:wkernel+(j*sw)],
                               axis=(1, 2))
            pooled[mImage, i, j] = data

    return pooled
