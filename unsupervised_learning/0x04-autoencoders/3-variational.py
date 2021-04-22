#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:09:29 2021

@author: Robinson Montes
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder

    Arguments:
     - input_dims is an integer containing the dimensions of the model input
     - hidden_layers is a list containing the number of nodes for each
        hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
     - latent_dims is an integer containing the dimensions of the latent
        space representation

    Returns:
     encoder, decoder, auto
        - encoder is the encoder model, which should output
            the latent representation, the mean, and the log variance,
            respectively
        - decoder is the decoder model
        - auto is the full autoencoder model
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims, ))

    hidden_layer = keras.layers.Dense(hidden_layers[0],
                                      activation='relu')(input_encoder)

    for i in range(1, len(hidden_layers)):
        hidden_layer = keras.layers.Dense(hidden_layers[i],
                                          activation='relu')(hidden_layer)

    z_mean = keras.layers.Dense(latent_dims)(hidden_layer)
    z_var = keras.layers.Dense(latent_dims)(hidden_layer)

    def sampling(args):
        """
        Sampling the data from the data set using the z_mean and z_stand_dev
        """
        z_mean, z_var = args
        m = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(m, dims))

        return z_mean + keras.backend.exp(0.5 * z_var) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_var])

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims, ))
    hidden_layer = keras.layers.Dense(hidden_layers[-1],
                                      activation='relu')(input_decoder)

    for i in range(len(hidden_layers)-2, -1, -1):
        hidden_layer = keras.layers.Dense(hidden_layers[i],
                                          activation='relu')(hidden_layer)
    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(hidden_layer)

    encoder = keras.models.Model(inputs=input_encoder,
                                 outputs=[z, z_mean, z_var])
    decoder = keras.models.Model(inputs=input_decoder,
                                 outputs=output_decoder)

    out_encoder = encoder(input_encoder)[0]
    out_decoder = decoder(out_encoder)
    # Autoencoder
    autoencoder = keras.models.Model(inputs=input_encoder,
                                     outputs=out_decoder)

    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    def loss(y_in, y_out):
        """
        Custom loss function
        """
        y_loss = keras.backend.binary_crossentropy(y_in, y_out)
        y_loss = keras.backend.sum(y_loss, axis=1)
        kl_loss = (1 + z_var - keras.backend.square(z_mean) -
                   keras.backend.exp(z_var))
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=1)

        return y_loss + kl_loss

    autoencoder.compile(optimizer='Adam', loss=loss)

    return encoder, decoder, autoencoder
