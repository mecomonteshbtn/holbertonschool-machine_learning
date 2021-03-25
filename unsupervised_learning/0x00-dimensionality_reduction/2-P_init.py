#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np


def P_init(X, perplexity):
    """
    Function that initializes all variables required to calculate
    the P affinities in t-SNE

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the dataset
        to be transformed by t-SNE
        * n is the number of data points
        * d is the number of dimensions in each point
     - perplexity is the perplexity that all Gaussian distributions should have

    Returns:
     (D, P, betas, H)
        - D: a numpy.ndarray of shape (n, n) that calculates
            the squared pairwise distance between two data points
        - P: a numpy.ndarray of shape (n, n) initialized to all 0‘s
            that will contain the P affinities
        - betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s
            that will contain all of the beta values
            [beta{i} = 1/(2 * sigma{i}^2)]
        - H is the Shannon entropy for perplexity perplexity
    """

    n, d = X.shape

    EX = np.sum(np.square(X), axis=1)
    D = (np.add(np.add(-2 * np.dot(X, X.T), EX).T, EX))
    D[range(n), range(n)] = 0
    P = np.zeros([n, n], dtype='float64')
    betas = np.ones([n, 1], dtype='float64')
    H = np.log2(perplexity)

    return (D, P, betas, H)
