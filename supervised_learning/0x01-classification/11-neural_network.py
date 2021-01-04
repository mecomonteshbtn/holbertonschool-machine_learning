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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
          * nx is the number of input features to the neuron
          * m is the number of examples
        Updates the private attributes __A1 and __A2
        The neurons should use a sigmoid activation function
        Returns:
         The private attributes __A1 and __A2, respectively
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)

        return self.__A1, self.__A2

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
         - A (numpy.ndarray): is a numpy.ndarray with shape (1, m) containing
           the activated output of the neuron for each example
        Returns:
         The cost
        """
        y1 = 1 - Y
        y2 = 1.0000001 - A
        m = Y.shape[1]

        cost = -np.sum(Y * np.log(A) + y1 * np.log(y2)) / m

        return cost
