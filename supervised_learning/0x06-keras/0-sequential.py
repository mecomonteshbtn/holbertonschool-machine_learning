#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that that builds a neural network with the Keras library
    Arguments:
     - nx is the number of input features to the network
     - layers is a list containing the number of nodes in each layer
        of the network
     - activations is a list containing the activation functions used for
        each layer of the network
     - lambtha is the L2 regularization parameter
     - keep_prob is the probability that a node will be kept for dropout
    Returns:
     The keras model
    """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=regularizer,
                             name='input_layer'))

    for layer in range(1, len(layers)):
        dname = 'layer_' + str(layer)
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[layer],
                                 activation=activations[layer],
                                 kernel_regularizer=regularizer,
                                 name=dname))

    return model
