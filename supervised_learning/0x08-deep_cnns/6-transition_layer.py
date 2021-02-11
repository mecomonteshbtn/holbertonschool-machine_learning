#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer as described
    in Densely Connected Convolutional Networks:

    Arguments:
     - X is the output from the previous layer
     - nb_filters is an integer representing the number of filters in X
      compression is the compression factor for the transition layer

    Returns:
     The output of the transition layer
     and the number of filters within the output, respectively
    """

    kernel_init = K.initializers.he_normal(seed=None)
    nfc = int(nb_filters * compression)

    batchNorm1 = K.layers.BatchNormalization()(X)
    activation1 = K.layers.Activation('relu')(batchNorm1)
    tlayer = K.layers.Conv2D(filters=nfc, kernel_size=1, padding='same',
                             kernel_initializer=kernel_init)(activation1)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='same')(tlayer)

    return avg_pool, nfc
