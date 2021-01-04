#!/usr/bin/env python3
"""
Class NeuralNetwork
"""


import numpy as np
import matplotlib.pyplot as plt


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

        # public intance attributes
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
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

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

        cost = -1 * (1 / m) * np.sum(Y * np.log(A) + y1 * np.log(y2))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
            labels for the input data
        Returns:
         The neuron’s prediction and the cost of the network, respectively
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)

        return (np.round(A2).astype(int), cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
           labels for the input data
         - A1 is the output of the hidden layer
         - A2 is the predicted output
         - alpha is the learning rate
        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(A1, dZ2.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        g1 = A1 * (1 - A1)
        dZ1 = np.multiply(np.matmul(self.__W2.T, dZ2), g1)
        dW1 = (1 / m) * np.matmul(X, dZ1.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - alpha * db2

        self.__W1 = self.__W1 - (alpha * dW1).T
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
           labels for the input data
         - iterations (int): is the number of iterations to train over
         - alpha (float): is the learning rate
         Returns:
          The evaluation of the training data after iterations of training
          have occurred
          Updates the private attributes __W1, __b1, __A1, __W2, __b2, & __A2
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            cost = self.cost(Y, self.__A2)

            if verbose:
                if i % step == 0 or step == iterations:
                    cost_list.append(cost)
                    step_list.append(i)
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)
