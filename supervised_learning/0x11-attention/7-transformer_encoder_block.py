#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class to create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        Arguments:
         - dm - the dimensionality of the model
         - h - the number of heads
         - hidden - the number of hidden units in the fully connected layer
         - drop_rate - the dropout rate

        Public instance attributes:
         - mha - a MultiHeadAttention layer
         - dense_hidden - the hidden dense layer with hidden units
         and relu activation
         - dense_output - the output dense layer with dm units
         - layernorm1 - the first layer norm layer, with epsilon=1e-6
         - layernorm2 - the second layer norm layer, with epsilon=1e-6
         - dropout1 - the first dropout layer
         - dropout2 - the second dropout layer
        """

        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Public instance method
        Arguments:
         - x - a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder block
         - training - a boolean to determine if the model is training
         - mask - the mask to be applied for multi head attention

        Returns:
         A tensor of shape (batch, input_seq_len, dm)
         containing the block’s output
        """

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
