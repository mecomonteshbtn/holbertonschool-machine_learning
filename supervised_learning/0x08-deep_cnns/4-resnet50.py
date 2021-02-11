#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 11:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)

    Returns:
     The keras model
    """

    activation = 'relu'
    kernel_init = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    layer1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', kernel_initializer=kernel_init)(X)
    batchNorm_l1 = K.layers.BatchNormalization(axis=3)(layer1)
    activation1 = K.layers.Activation('relu')(batchNorm_l1)
    layer_pool1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(activation1)

    layer2 = projection_block(layer_pool1, [64, 64, 256], 1)
    layer3 = identity_block(layer2, [64, 64, 256])
    layer4 = identity_block(layer3, [64, 64, 256])
    layer5 = projection_block(layer4, [128, 128, 512])
    layer6 = identity_block(layer5, [128, 128, 512])
    layer7 = identity_block(layer6, [128, 128, 512])
    layer8 = identity_block(layer7, [128, 128, 512])
    layer9 = projection_block(layer8, [256, 256, 1024])
    layer10 = identity_block(layer9, [256, 256, 1024])
    layer11 = identity_block(layer10, [256, 256, 1024])
    layer12 = identity_block(layer11, [256, 256, 1024])
    layer13 = identity_block(layer12, [256, 256, 1024])
    layer14 = identity_block(layer13, [256, 256, 1024])
    layer15 = projection_block(layer14, [512, 512, 2048])
    layer16 = identity_block(layer15, [512, 512, 2048])
    layer17 = identity_block(layer16, [512, 512, 2048])

    average_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                             padding='same')(layer17)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(average_pool)
    model = K.models.Model(inputs=X, outputs=Y)

    return model
