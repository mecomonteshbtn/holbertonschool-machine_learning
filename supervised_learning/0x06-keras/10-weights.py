#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Function saves a model’s weights

    Arguments:
     - network is the model whose weights should be saved
     - filename is the path of the file that the weights should be saved to
     - save_format is the format in which the weights should be saved

    Returns:
     None
    """
    network.save_weights(filename)

    return None


def load_weights(network, filename):
    """
    Function that loads a model’s weights
    Arguments:
     - network is the model to which the weights should be loaded
     - filename is the path of the file that the weights should be loaded from

    Returns:
     None
    """
    network.load_weights(filename)

    return None
