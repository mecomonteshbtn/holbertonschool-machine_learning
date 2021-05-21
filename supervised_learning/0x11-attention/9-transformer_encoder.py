#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class to create the encoder for a transformer
    """

    def __init__(self, N, dm, h,
                 hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Arguments:
         - N - the number of blocks in the encoder
         - dm - the dimensionality of the model
         - h - the number of heads
         - hidden - the number of hidden units in the fully connected layer
         - input_vocab - the size of the input vocabulary
         - max_seq_len - the maximum sequence length possible
         - drop_rate - the dropout rate

        Public instance attributes:
         - N - the number of blocks in the encoder
         - dm - the dimensionality of the model
         - embedding - the embedding layer for the inputs
         - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
         containing the positional encodings
         - blocks - a list of length N containing all of the EncoderBlock‘s
         - dropout - the dropout layer, to be applied to
            the positional encodings
        """

        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for n in range(self.N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Public instance method

        Arguments:
         - x - a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder
         - training - a boolean to determine if the model is training
         - mask - the mask to be applied for multi head attention
        Returns:
         A tensor of shape (batch, input_seq_len, dm)
         containing the encoder output
        """

        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        out = self.dropout(x, training=training)
        for n in range(self.N):
            out = self.blocks[n](out, training, mask)

        return out
