#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 06:03:48 2021

@author: Robinson Montes
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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
     path, P, or None, None on failure
        * path is the a list of length T containing the most likely sequence
            of hidden states
        * P is the probability of obtaining the path sequence
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

    F = np.empty((N, T))
    B = np.empty((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    B[:, 0] = 0

    for t in range(1, T):
        prob = (F[:, t - 1] * Transition.T *
                Emission[np.newaxis, :, Observation[t]].T)
        F[:, t] = np.amax(prob, 1)
        B[:, t - 1] = np.argmax(prob, 1)

    path = [0] * T
    last = np.argmax(F[:, T - 1])
    path[0] = last
    idx = 1

    for i in range(T - 2, -1, -1):
        path[idx] = int(B[int(last), i])
        last = B[int(last), i]
        idx += 1

    path.reverse()
    P = np.amax(F, axis=0)
    P = np.amin(P)

    return path, P
