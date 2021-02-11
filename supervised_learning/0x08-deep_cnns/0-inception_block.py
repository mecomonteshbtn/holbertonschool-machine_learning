#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Function that that builds an inception block
    as described in Going Deeper with Convolutions (2014).

    Arguments:
     - A_prev is the output from the previous layer
     - filters is a tuple or list containing:
        * F1 is the number of filters in the 1x1 convolution
        * F3R is the number of filters in the 1x1 convolution before
            the 3x3 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F5R is the number of filters in the 1x1 convolution before
            the 5x5 convolution
        * F5 is the number of filters in the 5x5 convolution
        * FPP is the number of filters in the 1x1 convolution after
            the max pooling

    Returns:
     The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)

    layer_1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                              padding='same',
                              activation=activation,
                              kernel_initializer=kernel_init)(A_prev)

    layer_2R = K.layers.Conv2D(filters=F3R, kernel_size=1,
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_init)(A_prev)

    layer_2 = K.layers.Conv2D(filters=F3, kernel_size=3,
                              padding='same',
                              activation=activation,
                              kernel_initializer=kernel_init)(layer_2R)

    layer_3R = K.layers.Conv2D(filters=F5R, kernel_size=1,
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_init)(A_prev)

    layer_3 = K.layers.Conv2D(filters=F5, kernel_size=5,
                              padding='same',
                              activation=activation,
                              kernel_initializer=kernel_init)(layer_3R)

    layer_pool = K.layers.MaxPooling2D(pool_size=[3, 3],
                                       strides=1,
                                       padding='same')(A_prev)

    layer_poolR = K.layers.Conv2D(filters=FPP, kernel_size=1,
                                  padding='same',
                                  activation=activation,
                                  kernel_initializer=kernel_init)(layer_pool)

    layers_list = [layer_1, layer_2, layer_3, layer_poolR]
    concatenated = K.layers.concatenate(layers_list)
    return concatenated
