#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function that tests a neural network

    Arguments:
     - network is the network model to test
     - data is the input data to test the model with
     - labels are the correct one-hot labels of data
     - verbose is a boolean that determines if output should be printed
        during the testing process

    Returns:
     The loss and accuracy of the model with the testing data, respectively
    """
    test_nn = network.evaluate(data, labels, verbose=verbose)

    return test_nn
