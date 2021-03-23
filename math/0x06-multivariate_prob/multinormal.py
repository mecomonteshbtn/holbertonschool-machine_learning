#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:45:46 2021

@author: Robinson Montes
"""
import numpy as np


class MultiNormal(object):
    """
    Class that represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Constructor funtcion

        Arguments:
         - data is a numpy.ndarray of shape (d, n) containing the data set:
         - n is the number of data points
         - d is the number of dimensions in each data point

        Public instance variables:
         - mean - a numpy.ndarray of shape (d, 1) containing the mean of data
         - cov - a numpy.ndarray of shape (d, d) containing
            the covariance matrix data
        """

        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')

        d = data.shape[0]
        self.mean = (np.mean(data, axis=1)).reshape(d, 1)

        X = data - self.mean
        self.cov = (np.dot(X, X.T)) / (n - 1)

    # Public instance method
    def pdf(self, x):
        """
        Public instance method def pdf that calculates the PDF at a data point

        Arguments:
         - x is a numpy.ndarray of shape (d, 1) containing the data point
            whose PDF should be calculated
            * d is the number of dimensions of the Multinomial instance

        Returns
         The value of the PDF
        """

        if type(x) != np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        n = x.shape[0]

        m = self.mean
        c = self.cov

        den = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(c))
        icov = np.linalg.inv(c)
        expo = (-0.5 * np.matmul(np.matmul((x - m).T, icov), x - self.mean))

        pdf = (1 / den) * np.exp(expo[0][0])

        return pdf
