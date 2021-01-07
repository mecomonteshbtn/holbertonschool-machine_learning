#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:51:37 2021

@author: Robinson Montes
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Function create_layer
    Arguments:
     - prev: is the tensor output of the previous layer
     - n: is the number of nodes in the layer to create
     - activation: is the activation function that the layer should use
    Returns:
    The tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')

    return layer(prev)
