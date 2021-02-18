#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K
import numpy as np


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)

    train_datagen = K.preprocessing.image.ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_gen = train_datagen.flow(x_train, y_train, batch_size=30)

    x_test = K.applications.densenet.preprocess_input(x_test)
    y_test = K.utils.to_categorical(y_test, 10)

    inputs = K.Input(shape=(32, 32, 3))
    inputs = K.layers.UpSampling2D()(inputs)

    network = K.applications.densenet.DenseNet121(include_top=False,
                                                  pooling='max',
                                                  input_tensor=inputs,
                                                  weights='imagenet')

    output = network.layers[-1].output
    output = K.layers.Flatten()(output)
    output = K.layers.Dense(512, activation='relu')(output)
    output = K.layers.Dropout(0.15)(output)
    output = K.layers.Dense(256, activation='relu')(output)
    output = K.layers.Dropout(0.15)(output)
    output = K.layers.Dense(10, activation='softmax')(output)

    model = K.models.Model(network.input, output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return 0.001 / (1 + 1 * epoch)

    callbacks = []

    lrr__callback = K.callbacks.LearningRateScheduler(learning_rate,
                                                      verbose=1)
    callbacks.append(lrr__callback)

    es__callback = K.callbacks.EarlyStopping(monitor='val_acc',
                                             mode='max',
                                             verbose=1,
                                             patience=5)
    callbacks.append(es__callback)

    mc__callback = K.callbacks.ModelCheckpoint('cifar10.h5',
                                               monitor='val_acc',
                                               mode='max',
                                               verbose=1,
                                               save_best_only=True)
    callbacks.append(mc__callback)

    history = model.fit(train_gen,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        callbacks=callbacks,
                        epochs=32,
                        verbose=1)


def preprocess_data(X, Y):
    """
    pre-processes the data for your model
    :param X: numpy.ndarray of shape (m, 32, 32, 3)
        containing the CIFAR 10 data,
        where m is the number of data points
    :param Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p
