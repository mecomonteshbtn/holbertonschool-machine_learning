#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:12:16 2020

@author: meco
"""


class Exponential:
    """
    Representing an Exponential distribution
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Arguments:
        - data (list): is a list of the data to be used to estimate the
        distribution
        - lambtha (int/float): is the expected number of occurences
        in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Arguments:
         - x (int): is the time period
        Return:
         The PDF value for x
        """
        if x < 0:
            return 0

        pdf = self.lambtha * pow(Exponential.e, -self.lambtha * x)

        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period

        Arguments:
         - x (int): is the time period
        Return:
         The CDF value for x
        """
        if x < 0:
            return 0

        cdf = 1 - pow(Exponential.e, -self.lambtha * x)

        return cdf
