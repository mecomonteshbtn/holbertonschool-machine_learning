#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:00:22 2021

@author: Robinson Montes
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function that converts a label vector into a one-hot matrix
    Arguments:
     - labels
     - classes is the number of classes
    Returns:
     The one-hot matrix
    """
    encoded = K.utils.to_categorical(labels, classes)

    return encoded
