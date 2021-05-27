#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow.compat.v2 as tf
sdp_attention = __import__('sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform multi head attention
    """

    def __init__(self, dm, h):
        """
        Class constructor

        Arguments:
         - dm is an integer representing the dimensionality of the model
         - h is an integer representing the number of heads

        Public instance attributes:
         - h - the number of heads
         - dm - the dimensionality of the model
         - depth - the depth of each attention head
         - Wq - a Dense layer with dm units, used to generate the query matrix
         - Wk - a Dense layer with dm units, used to generate the key matrix
         - Wv - a Dense layer with dm units, used to generate the value matrix
         - linear - a Dense layer with dm units, used to generate
            the attention output
        """

        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm // self.h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def splitHeads(self, m, batch):
        """
        Function to split last dim shape(self.h, self.depth)
        Return:
         transpose result shape(batch, -1, self.h, self.depth)
        """
        m = tf.reshape(m, (batch, -1, self.h, self.depth))
        return tf.transpose(m, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Public instance method

        Arguments:
         - Q is a tensor of shape (batch, seq_len_q, dk)
            containing the input to generate the query matrix
         - K is a tensor of shape (batch, seq_len_v, dk)
            containing the input to generate the key matrix
         - V is a tensor of shape (batch, seq_len_v, dv)
            containing the input to generate the value matrix
         - mask is always None

        Returns:
         output, weights
         - output a tensor with its last two dimensions as (..., seq_len_q, dm)
            containing the scaled dot product attention
         - weights a tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) containing the attention weights
        """

        batch = tf.shape(K)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split dims(self.h, batch)
        Q = self.splitHeads(Q, batch)
        K = self.splitHeads(K, batch)
        V = self.splitHeads(V, batch)

        output, weights = sdp_attention(Q, K, V, mask)

        # Transpose output, reshape output,
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        output = self.linear(output)

        return output, weights
