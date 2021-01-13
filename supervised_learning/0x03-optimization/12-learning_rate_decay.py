#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow
    using inverse time decay
    Arguments:
     - alpha is the original learning rate
     - decay_rate is the weight used to determine the rate at which alpha
        will decay
     - global_step is the number of passes of gradient descent that
        have elapsed
     - decay_step is the number of passes of GD that should occur before
        alpha is decayed further
    Returns:
     The learning rate decay operation
    """

    decay = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)

    return decay
