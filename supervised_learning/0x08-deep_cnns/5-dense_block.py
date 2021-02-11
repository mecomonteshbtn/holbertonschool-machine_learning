#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block as described
    in Densely Connected Convolutional Networks:

    Arguments:
     - X is the output from the previous layer
     - nb_filters is an integer representing the number of filters in X
     - growth_rate is the growth rate for the dense block
     - layers is the number of layers in the dense block

    Returns:
     The concatenated output of each layer within the Dense Block
     and the number of filters within the concatenated outputs, respectively
    """

    kernel_init = K.initializers.he_normal(seed=None)

    for layer in range(layers):
        batchNorm1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(batchNorm1)
        conv1x1 = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                  padding='same',
                                  kernel_initializer=kernel_init)(activation1)

        batchNorm2 = K.layers.BatchNormalization()(conv1x1)
        activation2 = K.layers.Activation('relu')(batchNorm2)
        conv3x3 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                  padding='same',
                                  kernel_initializer=kernel_init)(activation2)

        X = K.layers.concatenate([X, conv3x3])
        nb_filters += growth_rate

    return X, nb_filters
