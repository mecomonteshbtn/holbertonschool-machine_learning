#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:58:30 2020

@author: Robinson Montes
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Arguments:
         - X (numpy.ndarray): is a numpy.ndarray with shape (nx, m) that
           contains the input data
            *nx is the number of input features to the neuron
            *m is the number of examples
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data
        Returns:
         The neuron’s prediction and the cost of the network, respectively
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        return (np.round(A).astype(int), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Arguments:
         - X (numpy.ndarray): is a numpy.ndarray with shape (nx, m) that
           contains the input data
            *nx is the number of input features to the neuron
            *m is the number of examples
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data
         - A (numpy.ndarray): with shape (1, m) containing the activated output
            of the neuron for each example
         - alpha (float): is the learning rate
        Updates the private attributes __W & __b
        """
        dZ = A - Y
        m = Y.shape[1]
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron
        Arguments:
         - X (numpy.ndarray): is a numpy.ndarray with shape (nx, m) that
           contains the input data
            *nx is the number of input features to the neuron
            *m is the number of examples
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
            contains the correct labels for the input data
         - iterations (int): number of iterations to train over
         - alpha (float): is the learning rate
         - verbose (boolean): that defines whether or not to print information
           about the training. If True, print
            Cost after {iteration} iterations: {cost} every step iterations
         - graph (boolean): that defines whether or not to graph information
           about the training once the training has completed.
        Updates the private attributes __W, __b & __A
        Returns:
         The evaluation of the training data after iterations of training
         have occurred
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            cost = self.cost(Y, self.__A)
            cost_list.append(cost)
            step_list.append(i)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)
