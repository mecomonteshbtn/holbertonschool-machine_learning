#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Function that builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015):

    Arguments:
     - A_prev is the output from the previous layer
     - filters is a tuple or list containing:
        * F11 is the number of filters in the first 1x1 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F12 is the number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection
     - s is the stride of the first convolution in both
            the main path and the shortcut connection

    Returns:
     The activated output of the projection block
    """

    F11, F3, F12 = filters
    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)

    layer1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                             padding='same',
                             kernel_initializer=kernel_init)(A_prev)
    bathcNorm_l1 = K.layers.BatchNormalization(axis=3)(layer1)
    activation1 = K.layers.Activation('relu')(bathcNorm_l1)

    layer2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=kernel_init)(activation1)
    bathcNorm_l2 = K.layers.BatchNormalization(axis=3)(layer2)
    activation2 = K.layers.Activation('relu')(bathcNorm_l2)

    layer3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=kernel_init)(activation2)
    bathcNorm_l3 = K.layers.BatchNormalization(axis=3)(layer3)

    layer4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(s, s),
                             padding='same',
                             kernel_initializer=kernel_init)(A_prev)
    bathcNorm_l4 = K.layers.BatchNormalization(axis=3)(layer4)

    add = K.layers.Add()([bathcNorm_l3, bathcNorm_l4])
    activation3 = K.layers.Activation('relu')(add)

    return activation3
