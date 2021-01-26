#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Function that sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics
    Arguments:
     - network is the model to optimize
     - alpha is the learning rate
     - beta1 is the first Adam optimization parameter
     - beta2 is the second Adam optimization parameter
    Returns:
     None
    """
    adam = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=adam, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
