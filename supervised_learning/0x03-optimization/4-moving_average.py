#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""


def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set:
    Arguments:
     - data (list): is the list of data to calculate the moving average of
     - beta (list): is the weight used for the moving average
    Returns:
     A list containing the moving averages of data
    """

    moving_mean = []
    vt = 0
    t = 1

    for d in data:
        vt = beta * vt + (1 - beta) * d
        mt = vt / (1 - (beta ** t))
        moving_mean.append(mt)
        t += 1

    return moving_mean
