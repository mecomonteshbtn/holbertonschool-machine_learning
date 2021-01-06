#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 4 8:43:40 2021

@author: Robinson Montes
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

            z = np.matmul(W, Aprev) + b
            if i < self.__L - 1:
                self.__cache[Akey] = self.sigmoid(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

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

    def softmax(self, z):
        """
        Applies the softmax activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function

        Return:
        The private attribute y_hat
        """
        y_hat = np.exp(x - np.max(x))
        return y_hat / y_hat.sum(axis=0)

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
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """
        - Evaluates the neural network’s predictions
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        nx is the number of input features to the neuron and m is the number
        of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - Returns the neuron’s prediction and the cost of the network,
        respectively.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        # selects the max of each column, in a column vector
        Y_hat = np.max(A, axis=0)
        # put 1 in the max of each column, 0 otherwise
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        - Calculates one pass of gradient descent on the neural network.
        - Y is a numpy.ndarray with shape (1, m) that contains the
        correct labels for the input data.
        - cache is a dictionary containing all the intermediary
        values of the network.
        - alpha is the learning rate.
        - Updates the private attribute __weights.
        """
        # start the backpropagation
        m = Y.shape[1]
        # dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        weights_c = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                g = A * (1 - A)
                dz = (weights_c["W" + str(i + 1)].T @ dz) * g
            dw = (dz @ cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            # dz for next iteration
            self.__weights["W" + str(i)] = self.__weights[
                    "W" + str(i)] - (alpha * dw)
            self.__weights["b" + str(i)] = self.__weights[
                    "b" + str(i)] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        - Trains the deep neural network.
        - X is a numpy.ndarray with shape (nx, m)
        that contains the input data, nx is the number of input
        features to the neuron and m is the number of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data.
        - iterations is the number of iterations to train over.
        - alpha is the learning rate.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        cst = []
        it = []
        for epoc in range(0, iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            c = self.cost(Y, A)
            cst.append(c)
            it.append(epoc)
            if verbose and epoc % step == 0:
                print("Cost after {} iterations: {}".format(epoc, c))
        if verbose and (epoc + 1) % step == 0:
            print("Cost after {} iterations: {}".format(epoc + 1, c))
        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(it, cst, 'b-')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        - Saves the instance object to a file in pickle format.
        - filename is the file to which the object should be saved.
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        - Loads a pickled DeepNeuralNetwork object.
        - filename is the file from which the object should be loaded.
        Returns: the loaded object, or None if filename doesn’t exist
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
