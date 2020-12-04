#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:24:36 2020
@author: Robinson Montes
"""


def add_matrices(mat1, mat2):
    """
    Add two matrices.

    Parameters
    - matrix (list of lists): matrix to calculate the sum.

    Returns
     The sum of the 2 matrices as a list.
    """
    if np.shape(mat1) != np.shape(mat1):
        return None

    result = np.add(mat1, mat2)
    return result
