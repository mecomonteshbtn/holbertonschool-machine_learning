#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Function  saves an entire model
    Arguments:
     - network is the model to save
     - filename is the path of the file that the model should be saved to
    Returns:
     None
    """
    network.save(filename)

    return None


def load_model(filename):
    """
    Function that loads an entire model:
    Arguments:
     - filename is the path of the file that the model should be loaded from
    Returns:
     The loaded model
     """
    network = K.models.load_model(filename)

    return network
