#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np


def maximization(X, g):
    """
    Function that calculates the maximization step in the EM algorithm
    for a GMM

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the data set
     - g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster

    Returns:
     pi, m, S, or None, None, None on failure
        - pi is a numpy.ndarray of shape (k,) containing the updated priors
        for each cluster
        - m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means for each cluster
        - S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    prb = np.sum(g, axis=0)
    tot_prb = np.ones((n, ))
    if not np.isclose(prb, tot_prb).all():
        return None, None, None

    pi = np.zeros((k, ))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m_num = np.sum((g[i, :, np.newaxis] * X), axis=0)
        m_den = np.sum(g[i], axis=0)
        m[i] = m_num / m_den
        s_num = np.dot(g[i] * (X - m[i]).T, (X - m[i]))
        S[i] = s_num / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n

    return pi, m, S
