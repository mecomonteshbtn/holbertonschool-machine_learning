#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that finds the best number of clusters for a GMM using
    the Bayesian Information Criterion

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the data set
     - kmin is a positive integer containing the minimum number of clusters
        to check for (inclusive)
     - kmax is a positive integer containing the maximum number of clusters
        to check for (inclusive)
     - iterations is a positive integer containing the maximum number of
        iterations for the EM algorithm
     - tol is a non-negative float containing the tolerance for
        the EM algorithm
     - verbose is a boolean that determines if the EM algorithm should print
        information to the standard output

    Returns:
     best_k, best_result, l, b, or None, None, None, None on failure
        - best_k is the best value for k based on its BIC
        - best_result is tuple containing pi, m, S
              * pi is a numpy.ndarray of shape (k,) containing the cluster
                priors for the best number of clusters
              * m is a numpy.ndarray of shape (k, d) containing the centroid
                means for the best number of clusters
              * S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices for the best number of clusters
        - l is a numpy.ndarray of shape (kmax - kmin + 1) containing the
            log likelihood for each cluster size tested
        - b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested
            Use: BIC = p * ln(n) - 2 * l
            * p is the number of parameters required for the model
            * n is the number of data points used to create the model
            * l is the log likelihood of the model
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    n, d = X.shape
    k_r, result, l_b, b = [], [], [], []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, like = expectation_maximization(
            X, k, iterations, tol, verbose)
        k_r.append(k)
        result.append((pi, m, S))
        l_b.append(like)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * like
        b.append(bic)
    b = np.array(b)
    best = np.argmin(b)
    l_b = np.array(l_b)

    return k_r[best], result[best], l_b[best], b[best]
