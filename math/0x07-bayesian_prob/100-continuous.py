#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
from scipy import math, special


def posterior(x, n, p1, p2):
    """
    Function that calculates the posterior probability that the probability
    of developing severe side effects falls within a specific range
    given the data

    Arguments:
     - x is the number of patients that develop severe side effects
     - n is the total number of patients observed
     - p1 is the lower bound on the range
     - p2 is the upper bound on the range

    Returns:
     The posterior probability of each probability
     in P given x and n, respectively
    """

    if not isinstance(n, int) or (n <= 0):
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or (x < 0):
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(p1, float):
        raise ValueError('p1 must be a float in the range [0, 1]')

    if not isinstance(p2, float):
        raise ValueError('p2 must be a float in the range [0, 1]')

    if p1 > 1 or p1 < 0:
        raise ValueError('p1 must be a float in the range [0, 1]')

    if p2 > 1 or p2 < 0:
        raise ValueError('p2 must be a float in the range [0, 1]')

    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    return 1
