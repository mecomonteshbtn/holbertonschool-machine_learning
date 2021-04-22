#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:09:29 2021

@author: Robinson Montes
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional autoencoder

    Arguments:
     - input_dims is a tuple of integers containing the dimensions
        of the model input
     - filters is a list containing the number of filters for each
        convolutional layer in the encoder, respectively
        * the filters should be reversed for the decoder
     - latent_dims is a tuple of integers containing the dimensions
        of the latent space representation

    Returns:
     encoder, decoder, auto
         - encoder is the encoder model
         - decoder is the decoder model
         - auto is the full autoencoder model
    """
    # Encoder
    input_encoder = keras.Input(shape=input_dims)

    hidden_layer = keras.layers.Conv2D(filters=filters[0],
                                       kernel_size=3,
                                       padding='same',
                                       activation='relu')(input_encoder)
    output_encoder = keras.layers.MaxPool2D(pool_size=(2, 2),
                                            padding='same')(hidden_layer)

    for i in range(1, len(filters)):
        hidden_layer = keras.layers.Conv2D(filters=filters[i],
                                           kernel_size=3,
                                           padding='same',
                                           activation='relu')(hidden_layer)
        hidden_layer = keras.layers.MaxPool2D(pool_size=(2, 2),
                                              padding='same')(hidden_layer)

    output_encoder = hidden_layer

    # Decoder
    input_decoder = keras.Input(shape=latent_dims)

    hidden_layer = keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=3,
                                       padding='same',
                                       activation='relu')(input_decoder)
    hidden_layer = keras.layers.UpSampling2D(2)(hidden_layer)

    for i in range(len(filters)-2, 0, -1):
        hidden_layer = keras.layers.Conv2D(filters=filters[i],
                                           kernel_size=3,
                                           padding='same',
                                           activation='relu')(hidden_layer)
        hidden_layer = keras.layers.UpSampling2D(2)(hidden_layer)

    hidden_layer = keras.layers.Conv2D(filters=filters[0],
                                       kernel_size=3,
                                       padding='valid',
                                       activation='relu')(hidden_layer)
    hidden_layer = keras.layers.UpSampling2D(2)(hidden_layer)

    output_decoder = keras.layers.Conv2D(filters=input_dims[-1],
                                         kernel_size=3,
                                         padding='same',
                                         activation='sigmoid')(hidden_layer)

    encoder = keras.models.Model(inputs=input_encoder,
                                 outputs=output_encoder)
    decoder = keras.models.Model(inputs=input_decoder,
                                 outputs=output_decoder)

    encoder.summary()
    decoder.summary()
    input_autoencoder = keras.Input(shape=input_dims)
    out_encoder = encoder(input_autoencoder)
    out_decoder = decoder(out_encoder)

    # Autoencoder
    autoencoder = keras.models.Model(inputs=input_autoencoder,
                                     outputs=out_decoder)
    autoencoder.compile(optimizer='Adam',
                        loss='binary_crossentropy')

    return encoder, decoder, autoencoder
