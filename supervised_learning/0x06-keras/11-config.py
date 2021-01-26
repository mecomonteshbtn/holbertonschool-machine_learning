#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Function that saves a model’s configuration in JSON format

    Arguments:
     - network is the model whose configuration should be saved
     - filename is the path of the file that the configuration
        should be saved to

    Returns:
     None
    """
    json_model = network.to_json()
    with open(filename, 'w') as file:
        file.write(json_model)

    return None


def load_config(filename):
    """
    Function that loads a model with a specific configuration

    Arguments:
     - filename is the path of the file containing the model’s configuration
        in JSON format

    Returns:
     The loaded model
    """
    with open(filename, 'r') as file:
        json_model = K.models.model_from_json(file.read())

    return json_model
