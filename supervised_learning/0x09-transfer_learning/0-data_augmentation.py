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

# script should not run when the file is imported
if __name__ == '__main__':
    labels = ['airplane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    # dataset loading
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # pre-processing
    x_train = (x_train) / 255
    y_train = K.utils.to_categorical(y_train, 10)

    # data augmentation
    train_datagen = K.preprocessing.image.ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_gen = train_datagen.flow(x_train[14:15],
                                   y_train[14:15], batch_size=1)

    image = [next(train_gen) for i in range(0, 5)]
    fig, ax = plt.subplots(1, 5, figsize=(16, 6))
    print('Labels:', [item[1][0] for item in image])
    my_list = [ax[i].imshow(image[i][0][0]) for i in range(0, 5)]
    plt.show()
