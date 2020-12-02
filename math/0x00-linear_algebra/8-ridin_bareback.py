#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:24:36 2020

@author: Robinson Montes
"""


def mat_mul(mat1, mat2):
    """
    Function that performa matrix multiplication.

    Parameters:
    - mat1 (list of lists of ints/floats): 2D matrix to multiply.
    - mat2 (list of lists of ints/floats): 2D matrix to multiply.

    Return:
     A new matrix with the product of the matrices,
     if the two matrices cannot be multiplied, return None.
    """

    if len(mat1[0]) != len(mat2):
        return None

    m_matrix = []
    for r in range(len(mat1)):
        row = []
        for c in range(len(mat2[0])):
            i = 0
            for j in range(len(mat1[0])):
                k = mat1[r][j] * mat2[j][c]
                i += k
            row.append(i)
        m_matrix.append(row)

    return m_matrix
