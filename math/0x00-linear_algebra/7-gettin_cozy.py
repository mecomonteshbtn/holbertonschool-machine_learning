#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:24:36 2020

@author: Robinson Montes
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis.

    Parameters:
    - mat1 (list of lists of ints/floats): 2D matrix to concatenate.
    - mat2 (list of lists of ints/floats): 2D matrix to concatenate.
    - axix (int): axis to concatenate.

    Return:
     A new matrix that concatenates the matices,
     if the two matices cannot be concatenated, return None.
    """

    # verify if axis is 1 and the shape of the matrices
    if len(mat1) != len(mat2) and axis == 1:
        return None

    # verify if axis is 1 and the shape of the matrices
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
        return None

    c_matrix = []

    if axis == 0:
        c_matrix = mat1[:] + mat2[:]

    elif axis == 1:
        # Start to concatenate, first the rows
        for r in range(len(mat1)):
            rows = []
            # now the columns
            for c in range(len(mat1[0])):
                rows.append(mat1[r][c])
            for c in range(len(mat2[0])):
                rows.append(mat2[r][c])
            c_matrix.append(rows)

    return c_matrix
