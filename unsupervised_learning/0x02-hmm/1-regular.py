#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 06:03:48 2021

@author: Robinson Montes
"""
import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities of
    a regular markov chain:

    Arguments:
     - P is a is a square 2D numpy.ndarray of shape (n, n) representing
        the transition matrix
        * P[i, j] is the probability of transitioning from state i to state j
        * n is the number of states in the markov chain

    Returns:
     A numpy.ndarray of shape (1, n) containing the steady state probabilities,
     or None on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    s = np.ones((1, n1)) / n1

    while True:
        s_prev = s
        s = np.matmul(s, P)
        if np.any(P <= 0):
            return (None)
        if np.all(s_prev == s):
            return s
