#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that performs the expectation maximization for a GMM:

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the data set
     - k is a positive integer containing the number of clusters
     - iterations is a positive integer containing the maximum number
        of iterations for the algorithm
     - tol is a non-negative float containing tolerance of the log likelihood,
        used to determine early stopping i.e. if the difference is less than
        or equal to tol you should stop the algorithm
     - verbose is a boolean that determines if you should print information
        about the algorithm

    Returns:
     pi, m, S, g, l, or None, None, None, None, None on failure
     - pi is a numpy.ndarray of shape (k,) containing the priors for
        each cluster
     - m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
     - S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster
     - g is a numpy.ndarray of shape (k, n) containing the probabilities for
        each data point in each cluster
     - l is the log likelihood of the model
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    prev_like = 0
    g, likelihood = expectation(X, pi, m, S)

    for i in range(iterations):
        if verbose and (i % 10 == 0):
            msg = 'Log Likelihood after {} iterations: {}'\
                .format(i, likelihood.round(5))
            print(msg)
        pi, m, S = maximization(X, g)
        g, likelihood = expectation(X, pi, m, S)

        if abs(likelihood - prev_like) <= tol:
            break
        prev_like = likelihood

    if verbose:
        msg = 'Log Likelihood after {} iterations: {}'\
            .format(i + 1, likelihood.round(5))
        print(msg)

    return pi, m, S, g, likelihood
