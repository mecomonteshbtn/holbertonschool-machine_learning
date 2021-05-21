#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention

    Arguments:
    - Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    - K is a tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    - V is a tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    - mask is a tensor that can be broadcast into(..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None
        * if mask is not None, multiply -1e9 to the mask and add it to
         the scaled matrix multiplication

    Returns:
     output, weights
     - output a tensor with its last two dimensions as (..., seq_len_q, dv)
        containing the scaled dot product attention
     - weights tensor with its last two dimensions as
        (..., seq_len_q, seq_len_v) containing the attention weights
    """

    qk_dot = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    qk_scaled = qk_dot / tf.math.sqrt(dk)

    if mask is not None:
        qk_scaled += (mask * -1e9)

    weights = tf.nn.softmax(qk_scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
