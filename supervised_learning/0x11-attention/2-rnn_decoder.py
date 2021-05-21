#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class RNNDecoder to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Arguments:
         - vocab is an integer representing the size of the output vocabulary
         - embedding is an integer representing the dimensionality of
            the embedding vector
         - units is an integer representing the number of hidden units in
            the RNN cell
         - batch is an integer representing the batch size

        Public instance attributes:
         - embedding - a keras Embedding layer that converts words from
            the vocabulary into an embedding vector
         - gru - a keras GRU layer with units units
            * Should return both the full sequence of outputs as well as
                the last hidden state
            * Recurrent weights should be initialized with glorot_uniform
         - F - a Dense layer with vocab units
        """

        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Public instance method

        Arguments:
         - x is a tensor of shape (batch, 1)
            containing the previous word in the target sequence as
            an index of the target vocabulary
         - s_prev is a tensor of shape (batch, units)
            containing the previous decoder hidden state
         - hidden_states is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder

        Returns:
         y, s
         - y is a tensor of shape (batch, vocab)
            containing the output word as a one hot vector in
            the target vocabulary
         - s is a tensor of shape (batch, units)
            containing the new decoder hidden state
        """

        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        decode_outs, state = self.gru(x)

        decode_outs = tf.reshape(decode_outs, (-1, decode_outs.shape[2]))
        y = self.F(decode_outs)

        return y, state
