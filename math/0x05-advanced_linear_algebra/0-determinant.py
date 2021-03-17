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

    copy = list(map(list, matrix))
    dim = len(matrix)
    if dim == 1:
        return matrix[0][0]
    elif dim == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:
        for cur in range(dim):
            for i in range(cur + 1, dim):
                if copy[cur][cur] == 0:
                    copy[cur][cur] = 1.0e-18
                curScaler = copy[i][cur] / copy[cur][cur]
                for j in range(dim):
                    copy[i][j] = copy[i][j] - curScaler * copy[cur][j]
        det = 1.0
        for i in range(dim):
            det *= copy[i][i]
    return det
