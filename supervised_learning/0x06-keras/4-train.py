#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Function that trains a model using mini-batch gradient descent:
    Arguments:
     - network is the model to train
     - data is a numpy.ndarray of shape (m, nx) containing the input data
     - labels is a one-hot numpy.ndarray of shape (m, classes) containing
        the labels of data
     - batch_size is the size of the batch used for mini-batch gradient descent
     - epochs is the number of passes through data for mini-batch
        gradient descent
     - verbose is a boolean that determines if output should be printed
        during training
     - shuffle is a boolean that determines whether to shuffle the batches
        every epoch.
    Returns:
     The History object generated after training the model
    """
    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose)

    return history
