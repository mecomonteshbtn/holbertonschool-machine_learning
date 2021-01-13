#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Function that creates the training operation for a NN in tensorflow
    using the gradient descent with momentum optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the momentum weight
    Returns:
     The momentum optimization operation
    """

    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    optimized_momentum = optimizer.minimize(loss)

    return optimized_momentum
