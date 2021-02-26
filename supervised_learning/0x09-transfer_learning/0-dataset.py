#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:00:42 2021

@author: Robinson Montes
"""
import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    labels = ['airplane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)

    print(y_train[10:15])

    text_labels = [labels[int(i)] for i in y_train[10:15]]

    fig = plt.figure(figsize=(16, 6))
    for i in range(5):
        fig.add_subplot(1, 5, i + 1)
        plt.imshow(x_train[10 + i])
        plt.title(text_labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    sample_before = x_train[3]

    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)

    x_test = K.applications.densenet.preprocess_input(x_test)
    y_test = K.utils.to_categorical(y_test, 10)

    sample_after = x_train[3]

    print(sample_before[:, :, 0], '\n', sample_after[:, :, 0])

    print('before', sample_before[:, 0, 0], '\n')
    print('scale_factor', 255)
    print('scaled', sample_before[:, 0, 0] / 255, '\n')
    print('norm_factor', (sample_before[:, 0, 0] / 255) /
          sample_after[:, 0, 0])
    print('after', sample_after[:, 0, 0])
