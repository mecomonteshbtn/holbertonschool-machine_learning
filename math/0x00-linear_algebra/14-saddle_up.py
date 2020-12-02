#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:24:36 2020

@author: Robinson Montes
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Function that performs matrix multiplication

    Parameters:
    - mat1 (numpy.ndarray): never empty
    - mat2 (numpy.ndarray): never empty

    Return:
     The product of the matrices in ndarray
    """

    return np.matmul(mat1, mat2)
