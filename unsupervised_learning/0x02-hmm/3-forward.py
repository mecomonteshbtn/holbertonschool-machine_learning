#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 06:03:48 2021

@author: Robinson Montes
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm for a hidden markov model:

    Arguments:
     - Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation
        * T is the number of observations
     - Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
        * Emission[i, j] is the probability of observing j given
            the hidden state i
        * N is the number of hidden states
        * M is the number of all possible observations
     - Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
        * Transition[i, j] is the probability of transitioning from
            the hidden state i to j
     - Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state

    Returns:
     P, F, or None, None on failure
        * P is the likelihood of the observations given the model
        * F is a numpy.ndarray of shape (N, T) containing the forward
            path probabilities
            - F[i, j] is the probability of being in hidden state i at time j
                given the previous observations
    """

    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            F[i, t] = np.sum(F[:, t - 1] * Transition[:, i] *
                             Emission[i, Observation[t]])

    P = np.sum(F[:, -1:])

    return P, F
