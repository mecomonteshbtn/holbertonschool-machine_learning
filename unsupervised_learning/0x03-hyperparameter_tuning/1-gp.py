#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:03:21 2021

@author: Robinson Montes
"""
import numpy as np


class GaussianProcess():
    """
    Create the class GaussianProcess that represents a noiseless 1D Gaussian
    process:

    Class constructor: def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        - Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        - t is the number of initial samples
        - l is the length parameter for the kernel
        - sigma_f is the standard deviation given to the output of the
        black-box function
        - Sets the public instance attributes X, Y, l, and sigma_f
        corresponding to the respective constructor inputs
        - Sets the public instance attribute K, representing the current
        covariance kernel matrix for the Gaussian process

    Public instance method def kernel(self, X1, X2): that calculates the
    covariance kernel matrix between two matrices:
        - X1 is a numpy.ndarray of shape (m, 1)
        - X2 is a numpy.ndarray of shape (n, 1)
        - the kernel should use the Radial Basis Function (RBF)
    Returns:
    The covariance kernel matrix as a numpy.ndarray of shape (m, n)
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor

        Arguments:
         - X_init is a numpy.ndarray of shape (t, 1) representing
            the inputs already sampled with the black-box function
         - Y_init is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init
         - t is the number of initial samples
         - l is the length parameter for the kernel
         - sigma_f is the standard deviation given to the output
            of the black-box function

        Public instance attributes:
         - X corresponding to the respective constructor inputs
         - Y corresponding to the respective constructor inputs
         - l corresponding to the respective constructor inputs
         - sigma_f  corresponding to the respective constructor inputs
         - K  representing the current covariance kernel matrix
            for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Public instance method that calculates the covariance kernel matrix
        between two matrices

        Arguments:
         - X1 is a numpy.ndarray of shape (m, 1)
         - X2 is a numpy.ndarray of shape (n, 1)

        Returns:
         The covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 \
            * np.dot(X1, X2.T)
        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

        return K

    def predict(self, X_s):
        """
        Public instance method that predicts the mean
        and standard deviation of points in a Gaussian process

        Arguments:
         - X_s is a numpy.ndarray of shape (s, 1) containing all of the points
            whose mean and standard deviation should be calculated
            * s is the number of sample points

        Returns:
         mu, sigma
         - mu is a numpy.ndarray of shape (s,) containing the mean for
            each point in X_s, respectively
         - sigma is a numpy.ndarray of shape (s,) containing the
            standard deviation for each point in X_s, respectively
        """
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu = np.reshape(mu_s, -1)
        c_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diagonal(c_s)

        return mu, sigma
