#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 3 18:48:40 2021

@author: Robinson Montes
"""
import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork that defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features
         - nodes (int):  is the number of nodes found in the hidden layer
        Public instance attributes:
         - W1: The weights vector for the hidden layer.
               Upon instantiation, it should be initialized using a random
               normal distribution.
         - b1: The weights vector for the hidden layer. Upon instantiation,
               it should be initialized using a random normal distribution.
         - A1: The activated output for the hidden layer. Upon instantiation,
               it should be initialized to 0.
         - W2: The weights vector for the output neuron. Upon instantiation,
               it should be initialized using a random normal distribution.
         - b2: The bias for the output neuron. Upon instantiation, it should
               be initialized to 0.
         - A2: The activated output for the output neuron (prediction). Upon
               instantiation, it should be initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
