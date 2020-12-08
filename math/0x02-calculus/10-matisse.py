#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 9:34:16 2020

@author: Robinson Montes
"""


def poly_derivative(poly):
    """
     Function that find the derivate of a polynomial
    Arguments:
     - poly(list of integers): polynomial to calculate the derivate
    Return:
     List of coefficients representing the derivative of the polynomial
    """

    if poly is None or poly == [] or type(poly) is not list:
        return None

    derivate = []
    i = 0

    while i < len(poly):
        if type(poly[i]) not in (int, float):
            return None
        elif len(poly) == 1:
            derivate.append(0)
        else:
            if i == 0:
                i += 1
                continue
            derivate.append(poly[i]*i)
        i += 1

    return derivate
