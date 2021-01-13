#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function that updates a variable using the gradient descent with momentum
    optimization algorithm:
    Arguments:
     - alpha is the learning rate
     - beta1 is the momentum weight
     - var is a numpy.ndarray containing the variable to be updated
     - grad is a numpy.ndarray containing the gradient of var
     - v is the previous first moment of var
    Returns:
        The updated variable and the new moment, respectively
    """
    vt = np.multiply(beta1, v) + np.multiply((1 - beta1), grad)
    var_t = var - np.multiply(alpha, vt)

    return var_t, vt
