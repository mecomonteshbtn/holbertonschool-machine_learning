#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:51:37 2021

@author: Robinson Montes
"""
import tensorflow as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the NN
    Arguments:
     - x: is the placeholder for the input data.
     - layer_sizes: is a list containing the number of nodes in each layer of
      the network.
     _ activations: is a list containing the activation functions for each
      layer of the network.
    Returns:
    The prediction of the network in tensor form
    """
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer
