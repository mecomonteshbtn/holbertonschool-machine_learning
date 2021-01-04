#!/usr/bin/env python3
"""
Class DeepNeuralNetwork
"""


import numpy as np


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features to the neuron
         - layers (list): representing the number of nodes in each layer of
                          the network
        Public instance attributes:
         - L: The number of layers in the neural network.
         - cache: A dictionary to hold all intermediary values of the network.
         - weights: A dictionary to hold all weights and biased of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Private intance attributes
        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """
        getter function for L
        Returns the number of layers
        """
        return self.__L

    @property
    def cache(self):
        """
        getter gunction for cache
        Returns a dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter function for weights
        Returns a dictionary to hold all weights and biased of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            Z = np.matmul(W, Aprev) + b
            self.__cache[Akey] = 1 / (1 + np.exp(-Z))

        return (self.__cache[Akey], self.__cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
                              labels for the input data
         - A (numpy.ndarray): with shape (1, m) containing the activated output
                              of the neuron for each example
        Returns:
         The cost
        """
        y1 = 1 - Y
        y2 = 1.0000001 - A
        m = Y.shape[1]
        cost = -1 * (1 / m) * np.sum(Y * np.log(A) + y1 * np.log(y2))

        return cost
