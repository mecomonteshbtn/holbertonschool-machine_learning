#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:25:51 2020

@author: meco
"""


class Normal:
    """
    Representing an Normal distribution
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Normal class constructor

        Arguments:
        - data (list): is a list of the data to be used to estimate the
        distribution
        - mean (float): is the mean of the distribution
        - stddev (float): is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data) / len(data))
                suma = 0
                for i in data:
                    suma += pow(i - self.mean, 2)
                self.stddev = pow(suma / len(data), 1/2)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        Arguments:
         - x (int/float): is the x-value
        Return:
         The z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        Arguments:
         - z (int/float): is the z-score
        Return:
         The x-value of z
        """
        return self.mean + (self.stddev * z)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        Arguments:
         - x (int): is the x-value
        Return:
         The PDF value for x
        """
        variance = pow(self.stddev, 2)
        exp = -pow(x - self.mean, 2) / (2 * variance)
        sd = pow(2 * Normal.pi, 1 / 2) * self.stddev
        pdf = pow(Normal.e, exp) / sd

        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Arguments:
         - x (int): is the x-value
        Return:
         The CDF value for x
        """

        z = (x - self.mean) / (self.stddev * pow(2, 1 / 2))
        er = z - pow(z, 3) / 3 + pow(z, 5) / 10 - pow(z, 7) / 42 + pow(z,
                                                                       9) / 216

        cdf = 0.5 * (1 + (2 / (Normal.pi ** (1/2))) * (er))

        return cdf
