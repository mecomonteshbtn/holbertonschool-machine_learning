#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function that calculates the cost of a NN with L2 regularization
    Arguments:
     - cost is a tensor containing the cost of the network without
        L2 regularization
    Returns:
    A tensor containing the cost of the network accounting for
    L2 regularization
    """
    L2_cost = cost + tf.losses.get_regularization_losses()

    return L2_cost
