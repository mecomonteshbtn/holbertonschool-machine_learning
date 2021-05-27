#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:32:21 2021

@author: Robinson Montes
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer

    Arguments:
     - max_seq_len is an integer representing the maximum sequence length
     - dm is the model depth

    Returns:
     A numpy.ndarray of shape (max_seq_len, dm) containing
     the positional encoding vectors
    """

    p_encoding = np.zeros([max_seq_len, dm])

    for i in range(dm):
        for pos in range(max_seq_len):
            p_encoding[pos, i] = pos / np.power(10000, (2 * (i // 2) / dm))

    p_encoding[:, 0::2] = np.sin(p_encoding[:, 0::2])
    p_encoding[:, 1::2] = np.cos(p_encoding[:, 1::2])

    return p_encoding
