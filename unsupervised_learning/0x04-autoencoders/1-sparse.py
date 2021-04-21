#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:09:29 2021

@author: Robinson Montes
"""
import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Function that creates a sparse autoencoder

    Arguments:
     - input_dims is an integer containing the dimensions of the model input
     - hidden_layers is a list containing the number of nodes for each
        hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
     - latent_dims is an integer containing the dimensions of
        the latent space representation
     - lambtha is the regularization parameter used for L1 regularization
        on the encoded output

    Returns:
     encoder, decoder, auto
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the sparse autoencoder model
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims, ))

    hidden_encoder = keras.layers.Dense(hidden_layers[0],
                                        activation='relu', )(input_encoder)

    for i in range(1, len(hidden_layers)):
        hidden_encoder = keras.layers.Dense(hidden_layers[i],
                                            activation='relu')(hidden_encoder)
    reg = keras.regularizers.l1(lambtha)
    oencoder = keras.layers.Dense(latent_dims,
                                  activation='relu',
                                  activity_regularizer=reg)(hidden_encoder)

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims, ))
    hidden_decoder = keras.layers.Dense(hidden_layers[-1],
                                        activation='relu')(input_decoder)

    for i in range(len(hidden_layers)-2, -1, -1):
        hidden_decoder = keras.layers.Dense(hidden_layers[i],
                                            activation='relu')(hidden_decoder)
    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid',)(hidden_decoder)

    encoder = keras.models.Model(inputs=input_encoder, outputs=oencoder)
    decoder = keras.models.Model(inputs=input_decoder, outputs=output_decoder)

    out_encoder = encoder(input_encoder)
    out_decoder = decoder(out_encoder)

    # Autoencoder
    autoencoder = keras.models.Model(inputs=input_encoder,
                                     outputs=out_decoder)
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
