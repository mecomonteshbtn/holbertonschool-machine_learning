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

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function for W
        Returns weights
        """
        return self.__W

    @property
    def b(self):
        """
        getter gunction for b
        Returns bias
        """
        return self.__b

    @property
    def A(self):
        """
        getter function for A
        Returns activation values
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Arguments:
        - X (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute __A
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return self.__A

    def sigmoid(self, z):
        """
        Applies the sigmoid activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute A
        """
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
           contains the correct labels for the input data
         - A is a numpy.ndarray with shape (1, m) containing the activated
           output of the neuron for each example
        Returns:
         The cost
        """
        y1 = 1 - Y
        y2 = 1.0000001 - A
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + y1 * np.log(y2)) / m

        return cost
