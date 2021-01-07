#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:51:37 2021

@author: Robinson Montes
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network
    Arguments:
     - loss is the loss of the network’s prediction
     - alpha is the learning rate
    Returns:
    An operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    return optimizer
