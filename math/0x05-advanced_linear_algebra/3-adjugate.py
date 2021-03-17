#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 7:45:46 2021

@author: Robinson Montes
"""


def adjugate(matrix):
    """
    Function that calculates the adjugate matrix of a matrix:

    Arguments:
     - matrix is a list of lists whose adjugate matrix should be calculated

    Returns:
     The adjugate matrix of matrix
    """
    cof = cofactor(matrix)
    adj = []

    for r in range(len(matrix)):
        adj.append([])
        for c in range(len(matrix)):
            adj[r].append(cof[c][r])

    return adj


def cofactor(matrix):
    """
    Function that calculates the cofactor matrix of a matrix:

    Arguments:
     - matrix is a list of lists whose cofactor matrix should be calculated

    Returns:
     The cofactor matrix of matrix
    """
    cof = minor(matrix)
    for r in range(len(cof)):
        for c in range(len(cof[0])):
            cof[r][c] *= ((-1) ** (r + c))

    return cof


def submatrix(matrix, row, i):
    """
    auxiliar function to calculate a submatrix

    Arguments:
     - matrix: list of lists
     - row: rows
     - i: values
    """
    sub_m = []

    for r in row:
        aux = []
        for c in range(len(matrix)):
            if c != i:
                aux.append(r[c])
        sub_m.append(aux)

    return sub_m


def minor(matrix):
    """
    Function that calculates the minor matrix of a matrix:

    Aguments:
     - matrix is a list of lists whose minor matrix should be calculated

    Returns:
     The minor matrix of matrix
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not matrix:
        raise TypeError("matrix must be a list of lists")

    for r in matrix:
        if not isinstance(r, list):
            raise TypeError("matrix must be a list of lists")

    for r in matrix:
        if len(matrix) != len(r):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix[0]) == 1:
        return [[1]]

    minor_m = []
    for r in range(len(matrix)):
        min_r = []
        for c in range(len(matrix)):
            row = [matrix[i] for i in range(len(matrix)) if i != r]
            sub_m = submatrix(matrix, row, c)
            det = determinant(sub_m)
            min_r.append(det)
        minor_m.append(min_r)

    return minor_m


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
        sub_m = submatrix(matrix, row, i)
        det += j * (-1) ** i * determinant(sub_m)

    return det
