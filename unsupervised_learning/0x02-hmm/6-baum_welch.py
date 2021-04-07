#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 06:03:48 2021

@author: Robinson Montes
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for a hidden markov model

    Arguments:
     - Observations is a numpy.ndarray of shape (T,) that contains
        the index of the observation
        * T is the number of observations
     - Transition is a numpy.ndarray of shape (M, M) that contains
        the initialized transition probabilities
        * M is the number of hidden states
    - Emission is a numpy.ndarray of shape (M, N) that contains
        the initialized emission probabilities
        * N is the number of output states
    - Initial is a numpy.ndarray of shape (M, 1) that contains
        the initialized starting probabilities
     - iterations is the number of times expectation-maximization
        should be performed

    Returns:
     The converged Transition, Emission, or None, None on failure
    """

    if not isinstance(Observations, np.ndarray):
        return None, None

    if len(Observations.shape) != 1:
        return None, None

    T = Observations.shape[0]
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

    V = Observations
    b = Emission
    a = Transition
    M = N

    while True:
        P_forward, alpha = forward(Observations, Emission,
                                   Transition, Initial)
        alpha = alpha.T
        P_backward, beta = backward(Observations, Emission,
                                    Transition, Initial)
        beta = beta.T
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            d = np.dot(
                np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                n = alpha[t, i] * a[i, :] * \
                    b[:, V[t + 1]]. T * beta[t + 1, :].T
                xi[i, :, t] = n / d
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        K = b.shape[1]
        d = np.sum(gamma, axis=1)
        for i in range(K):
            b[:, i] = np.sum(gamma[:, V == 1], axis=1)

        b = np.divide(b, d.reshape((-1, 1)))
        if np.isclose(P_forward, P_backward):
            break

    return Transition, Emission


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

    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            F[i, t] = np.sum(F[:, t - 1] * Transition[:, i] *
                             Emission[i, Observation[t]])

    P = np.sum(F[:, -1:])

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    Function that performs the backward algorithm for a hidden markov model

    Arguments:
     - Observation is a numpy.ndarray of shape (T,) that contains
        the index of the observation
        * T is the number of observations
     - Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
        * Emission[i, j] is the probability of observing j given
            the hidden state i
        * N is the number of hidden states
        * M is the number of all possible observations
     - Transition is a 2D numpy.ndarray of shape (N, N) containing
        the transition probabilities
        * Transition[i, j] is the probability of transitioning from
            the hidden state i to j
     - Initial a numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state

    Returns:
     P, B, or None, None on failure
        - Pis the likelihood of the observations given the model
        - B is a numpy.ndarray of shape (N, T) containing
            the backward path probabilities
            * B[i, j] is the probability of generating the future observations
            from hidden state i at time j
    """

    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.zeros((N, T))
    B[:, T - 1] += 1

    for t in range(T - 2, -1, -1):
        for j in range(N):
            B[j, t] = np.sum(B[:, t + 1] * Transition[j, :] *
                             Emission[:, Observation[t + 1]])

    P = np.sum(B[:, 0] * Initial[:, 0] * Emission[:, Observation[0]])

    return P, B
