#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow.compat.v2 as tf
positional_encoding = __import__('positional_encoding').positional_encoding
DecoderBlock = __import__('transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class to create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Arguments:
         - N - the number of blocks in the encoder
         - dm - the dimensionality of the model
         - h - the number of heads
         - hidden - the number of hidden units in the fully connected layer
         - target_vocab - the size of the target vocabulary
         - max_seq_len - the maximum sequence length possible
         - drop_rate - the dropout rate

        Public instance attributes:
         - N - the number of blocks in the encoder
         - dm - the dimensionality of the model
         - embedding - the embedding layer for the targets
         - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
            containing the positional encodings
         - blocks - a list of length N containing all of the DecoderBlock‘s
         - dropout - the dropout layer, to be applied to
            the positional encodings
        """

        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Public instance method

        Arguments:
         - x - a tensor of shape (batch, target_seq_len, dm)
            containing the input to the decoder
         - encoder_output - a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
         - training - a boolean to determine if the model is training
         - look_ahead_mask - the mask to be applied to
            the first multi head attention layer
         - padding_mask - the mask to be applied to
            the second multi head attention layer

        Returns:
         A tensor of shape (batch, target_seq_len, dm)
         containing the decoder output
        """

        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for n in range(self.N):
            x = self.blocks[n](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x
