#!/usr/bin/env python3
"""
Create a Layer with Dropout
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Function that creates a layer of a NN using dropout:
    Arguments:
     - prev is a tensor containing the output of the previous layer
     - n is the number of nodes the new layer should contain
     - activation is the activation function that should be used on the layer
     - keep_prob is the probability that a node will be kept
    Returns:
     The output of the new layer
    """
    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=kernel_ini,
                            kernel_regularizer=kernel_reg)

    return layer(prev)
