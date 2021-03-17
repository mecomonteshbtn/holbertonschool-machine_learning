#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 7:45:46 2021

@author: Robinson Montes
"""


def determinant(matrix):
    """
    Function that calculates the determinant of a matrix:

    Arguments:
     - matrix is a list of lists whose determinant should be calculated

    Returns:
     The determinant of matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for r in matrix:
        if not isinstance(r, list):
            raise TypeError("matrix must be a list of lists")

    for r in matrix:
        if len(r) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return det

    det = 0
    for i, j in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        sub_m = []

        for r in row:
            aux = []
            for c in range(len(matrix)):
                if c != i:
                    aux.append(r[c])
            sub_m.append(aux)

        det += j * (-1) ** i * determinant(sub_m)

    return det
