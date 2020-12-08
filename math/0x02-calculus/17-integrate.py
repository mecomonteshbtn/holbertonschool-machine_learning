#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 9:34:16 2020

@author: Robinson Montes
"""


def poly_integral(poly, C=0):
    """
     Function that find the integral of a polynomial
    Arguments:
     - poly(list of integers): polynomial to calculate the derivate
     - C (int): constant of integration
    Return:
     List of coefficients representing the integral of the polynomial
    """

    if C is None or type(C) not in (int, float):
        return None

    if poly is None or poly == [] or type(poly) is not list:
        return None

    if poly == [0]:
        return [C]

    integrate = [C]
    i = 0

    while i < len(poly):
        if type(poly[i]) not in (int, float) or poly[i] is None:
            return None
        coef = poly[i] / (i + 1)
        if coef.is_integer():
            coef = int(coef)
        integrate.append(coef)
        i += 1

    return integrate
