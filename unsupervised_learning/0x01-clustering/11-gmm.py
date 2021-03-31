#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import sklearn.mixture


def gmm(X, k):
    """
    Function that calculates a GMM from a dataset

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the dataset
     - k is the number of clusters

    Returns:
     pi, m, S, clss, bic
        - pi is a numpy.ndarray of shape (k,) containing the cluster priors
        - m is a numpy.ndarray of shape (k, d) containing the centroid means
        - S is a numpy.ndarray of shape (k, d, d) containing
            the covariance matrices
        - clss is a numpy.ndarray of shape (n,) containing the cluster indices
            for each data point
        - bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested
    """

    mixture = sklearn.mixture.GaussianMixture(n_components=k)
    g = mixture.fit(X)
    m = g.means_
    S = g.covariances_
    pi = g.weights_
    clss = mixture.predict(X)
    bic = mixture.bic(X)

    return pi, m, S, clss, bic
