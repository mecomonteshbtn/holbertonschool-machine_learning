#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:24:36 2020

@author: Robinson Montes
"""


def add_arrays(arr1, arr2):
    """
    Function that add two arrays elemet-wise.

    Parameters:
    - arr1 (list of ints/floats): first list.
    - arr2 (list of ints/floats): second list.

    Return:
     A new list with the first list added second list,
     if the shape of the lists are not the same, return None.
    """

    # comparing the shape
    if len(arr1) != len(arr2):
        return None

    # the new list for the add of the lists
    add = []
    # cycle for take number by number
    for i in range(len(arr1)):
        add.append(arr1[i] + arr2[i])

    return add
