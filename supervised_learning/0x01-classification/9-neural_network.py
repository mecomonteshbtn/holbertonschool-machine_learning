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

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        getter function for W1
        Returns weights
        """
        return self.__W1

    @property
    def b1(self):
        """
        getter gunction for b1
        Returns bias
        """
        return self.__b1

    @property
    def A1(self):
        """
        getter function for A1
        Returns activation values
        """
        return self.__A1

    @property
    def W2(self):
        """
        getter function for W2
        Returns weights
        """
        return self.__W2

    @property
    def b2(self):
        """
        getter gunction for b2
        Returns bias
        """
        return self.__b2

    @property
    def A2(self):
        """
        getter function for A2
        Returns activation values
        """
        return self.__A2
