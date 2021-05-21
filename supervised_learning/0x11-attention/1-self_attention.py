#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Calculate the attention for machine translation
    """

    def __init__(self, units):
        """
        Class constructor

        Arguments:
         - units is an integer representing the number of hidden units
            in the alignment model

        Public instance attributes:
         - W - a Dense layer with units units, to be applied to
            the previous decoder hidden state
         - U - a Dense layer with units units, to be applied to
            the encoder hidden states
         - V - a Dense layer with 1 units, to be applied to
            the tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Public static method

        Arguments:
         - s_prev is a tensor of shape (batch, units)
            containing the previous decoder hidden state
         - hidden_states is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder

        Returns:
         context, weights
         - context is a tensor of shape (batch, units)
            that contains the context vector for the decoder
         - weights is a tensor of shape (batch, input_seq_len, 1)
            that contains the attention weights
        """

        Q = tf.expand_dims(s_prev, axis=1)
        w = self.W(Q)
        u = self.U(hidden_states)
        score = self.V(tf.nn.tanh(w + u))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum((weights * hidden_states), axis=1)

        return context, weights
