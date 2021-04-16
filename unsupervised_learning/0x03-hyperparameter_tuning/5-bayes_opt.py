#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:03:21 2021

@author: Robinson Montes
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Create the class BayesianOptimization that performs Bayesian optimization
    on a noiseless 1D Gaussian process:

    Class constructor def __init__(self, f, X_init, Y_init, bounds,
    ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        - f is the black-box function to be optimized
        - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        - Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        - t is the number of initial samples
        - bounds is a tuple of (min, max) representing the bounds of the spac
        in which to look for the optimal point
        - ac_samples is the number of samples that should be analyzed during
        acquisition
        - l is the length parameter for the kernel
        - sigma_f is the standard deviation given to the output of the
        black-box function
        - xsi is the exploration-exploitation factor for acquisition
        - minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False)
        - Sets the following public instance attributes:
            - f: the black-box function
            - gp: an instance of the class GaussianProcess
            - X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
            acquisition sample points, evenly spaced between min and max
            - xsi: the exploration-exploitation factor
            - minimize: a bool for minimization versus maximization
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Arguments:
            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing the inputs
            already sampled with the black-box function
            - Y_init is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init
            - t is the number of initial samples
            - bounds is a tuple of (min, max) representing the bounds of
            the space in which to look for the optimal point
            - ac_samples is the number of samples that should be
            analyzed during acquisition
            - l is the length parameter for the kernel
            - sigma_f is the standard deviation given to the output of
            the black-box function
            - xsi is the exploration-exploitation factor for acquisition
            - minimize is a bool determining whether optimization should be
            performed for minimization (True) or maximization (False)

        Public instance attributes:
            - f: the black-box function
            - gp: an instance of the class GaussianProcess
            - X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
            acquisition sample points, evenly spaced between min and max
            - xsi: the exploration-exploitation factor
            - minimize: a bool for minimization versus maximization
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Public instance method that calculates the next best sample location

        Uses the Expected Improvement acquisition function
        Returns:
         X_next, EI
            - X_next is a numpy.ndarray of shape (1,) representing
                the next best sample point
            - EI is a numpy.ndarray of shape (ac_samples,) containing
                the expected improvement of each potential sample
        """

        mu_sample, sigma_sample = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu_sample - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu_sample - Y_sample - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma_sample
            EI = (imp * norm.cdf(Z)) + (sigma_sample * norm.pdf(Z))
            EI[sigma_sample == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Public instance method that optimizes the black-box function

        Arguments:
         - iterations is the maximum number of iterations to perform

        Returns:
         X_opt, Y_opt
            - X_opt is a numpy.ndarray of shape (1,)
                representing the optimal point
            - Y_opt is a numpy.ndarray of shape (1,)
                representing the optimal function value
        """

        X_aux = []

        for i in range(iterations):
            X_opt, EI = self.acquisition()
            if X_opt in X_aux:
                break
            Y_opt = self.f(X_opt)
            self.gp.update(X_opt, Y_opt)
            X_aux.append(X_opt)

        if self.minimize is True:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
