#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:22:59 2020

@author: Robinson Montes
"""


class Binomial:
    """
    Representing an Binomial distribution
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor

        Arguments:
         - data (list): is a list of the data to be used to estimate the
        distribution
         - n (int): is the number of Bernoulli trials
         - p (float): is the probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / int(len(data))
                suma = 0
                for i in data:
                    suma += pow(i - mean, 2)
                var = suma / len(data)
                probability = 1 - var / mean
                self.n = int(round(mean / probability))
                self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PDF for a given number of “successes”

        Arguments:
         - k (int): is the number of “successes”
        Return:
         The PDF value for k
        """
        k = int(k)

        if k < 0:
            return 0

        n_fact = Binomial.factorial(self.n)
        k_fact = Binomial.factorial(k)
        nk_fact = Binomial.factorial(self.n - k)

        pmf = n_fact / (k_fact * nk_fact) * pow(self.p, k) * pow(1 - self.p,
                                                                 self.n - k)

        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”

        Arguments:
         - k (int): is the number of “successes”
        Return:
         The CDF value for k
        """

        if k < 0:
            return 0

        k = int(k)
        cdf = 0

        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf

    @classmethod
    def factorial(self, n):
        """
        Function to calculate the factorial of a number
        """
        fact = 1
        for x in range(1, n + 1):
            fact = x * fact
        return fact
