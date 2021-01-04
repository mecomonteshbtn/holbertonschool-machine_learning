#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:58:30 2020

@author: meco
"""
import numpy as np


class Neuron:
    """
    Class Neuron that defines a simple neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features to the neuron
        Public instance attributes:
         - W: The weights vector for the neuron. Upon instantiation, it should
              be initialized using a random normal distribution.
         - b: The bias for the neuron. Upon instantiation, it should be
              initialized to 0.
         - A: The activated output of the neuron (prediction). Upon
              instantiation, it should be initialized to 0.

        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
