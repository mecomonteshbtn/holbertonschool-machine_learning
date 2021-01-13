#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Function that creates the training operation for a NN in tensorflow
    using the RMSProp optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta2 is the RMSProp weight
     - epsilon is a small number to avoid division by zero
    Returns:
     The RMSProp optimization operation
    """

    optimizer = tf.train.RMSPropOptimizer(alpha, epsilon=epsilon, decay=beta2)
    optimized_RMSprop = optimizer.minimize(loss)

    return optimized_RMSprop
