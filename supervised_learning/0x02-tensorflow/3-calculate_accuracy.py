#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:51:37 2021

@author: Robinson Montes
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuranvy of a prediction.
    Arguments:
     - y: is a placeholder for the labels of the input data.
     - y_pred: is a tensor containing the network’s predictions.
    Returns:
    A tensor containing the decimal accuracy of the prediction.
    """
    eq = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))

    return accuracy
