#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:24:36 2020

@author: Robinson Montes
"""


def add_matrices2D(mat1, mat2):
    """
    Function that adds two matrices element-wise.

    Parameters:
    - mat1 (ints/floats list of lists): first matrix.
    - mat2 (ints/floats list of lists): second matrix.

    Return:
     A new matrix with first add second matrix,
     if mat1 and mat2 are not the same shape, return None.
    """

    if len(mat1) != len(mat2):
        return None

    add_matrix = []
    for r in range(len(mat1)):
        if len(mat1[r]) != len(mat2[r]):
            return None
        else:
            col = []
            for c in range(len(mat1[r])):
                col.append(mat1[r][c] + mat2[r][c])
            add_matrix.append(col)

    return add_matrix
