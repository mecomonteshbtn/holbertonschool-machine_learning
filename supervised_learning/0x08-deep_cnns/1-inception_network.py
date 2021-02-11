#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds the inception network as described
    in Going Deeper with Convolutions (2014)

    Returns:
     The keras model
    """

    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))

    l1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                         padding='same', activation=activation,
                         kernel_initializer=kernel_init)(X)

    l_pool1 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(l1)

    l2_1 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                           activation=activation,
                           kernel_initializer=kernel_init)(l_pool1)

    l2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                         activation=activation,
                         kernel_initializer=kernel_init)(l2_1)

    l_pool2 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(l2)

    l3 = inception_block(l_pool2, [64, 96, 128, 16, 32, 32])

    l4 = inception_block(l3, [128, 128, 192, 32, 96, 64])

    l_pool3 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(l4)

    l5 = inception_block(l_pool3, [192, 96, 208, 16, 48, 64])

    l6 = inception_block(l5, [160, 112, 224, 24, 64, 64])

    l7 = inception_block(l6, [128, 128, 256, 24, 64, 64])

    l8 = inception_block(l7, [112, 144, 288, 32, 64, 64])

    l9 = inception_block(l8, [256, 160, 320, 32, 128, 128])

    l_pool4 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                    padding='same')(l9)

    l10 = inception_block(l_pool4, [256, 160, 320, 32, 128, 128])

    l11 = inception_block(l10, [384, 192, 384, 48, 128, 128])

    l_avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7], strides=(7, 7),
                                           padding='same')(l11)

    dropout = K.layers.Dropout(0.4)(l_avg_pool)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(dropout)

    model = K.models.Model(inputs=X, outputs=Y)
    return model
