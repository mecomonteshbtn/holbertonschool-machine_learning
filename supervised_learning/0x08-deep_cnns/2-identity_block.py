#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Function that builds an identity block as described
    in Deep Residual Learning for Image Recognition (2015):

    Arguments:
     - A_prev is the output from the previous layer
     - filters is a tuple or list containing:
        * F11 is the number of filters in the first 1x1 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F12 is the number of filters in the second 1x1 convolution

    Returns:
     The activated output of the identity block
    """

    F11, F3, F12 = filters
    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)

    layer1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                             kernel_initializer=kernel_init)(A_prev)
    batchNorm_l1 = K.layers.BatchNormalization(axis=3)(layer1)
    activation1 = K.layers.Activation('relu')(batchNorm_l1)

    layer2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                             kernel_initializer=kernel_init)(activation1)
    batchNorm_l2 = K.layers.BatchNormalization(axis=3)(layer2)
    activation2 = K.layers.Activation('relu')(batchNorm_l2)

    layer3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                             kernel_initializer=kernel_init)(activation2)
    batchNorm_l3 = K.layers.BatchNormalization(axis=3)(layer3)
    add = K.layers.Add()([batchNorm_l3, A_prev])
    activation3 = K.layers.Activation('relu')(add)

    return activation3
