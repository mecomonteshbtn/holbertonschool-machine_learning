#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np


if __name__ == '__main__':
    labels = ['airplane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = (x_train) / 255
    y_train = K.utils.to_categorical(y_train, 10)

    inputs = K.Input(shape=(32, 32, 3))
    output = K.layers.UpSampling2D()(inputs)

    model = K.models.Model(inputs, output)

    model.summary()

    y_pred = model.predict(x_train[12:15])

    print(y_pred.shape)

    fig, ax = plt.subplots(1, 3, figsize=(32, 12))
    my_list = [ax[i].imshow(y_pred[i]) for i in range(0, 3)]
    plt.show()
