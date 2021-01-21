#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function that creates a tensorflow layer that includes L2 regularization
    Arguments:
     - prev is a tensor containing the output of the previous layer
     - n is the number of nodes the new layer should contain
     - activation is the activation function that should be used on the layer
     - lambtha is the L2 regularization parameter
    Returns:
     The output of the new layer
    """

    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=kernel_ini,
                            kernel_regularizer=kernel_reg)

    return layer(prev)
