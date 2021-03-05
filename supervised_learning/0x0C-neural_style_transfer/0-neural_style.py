#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:26:13 2021

@author: meco
"""
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class NST:
    """
    Create a class NST that performs tasks for neural style transfer:

    Public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'
    Class constructor: def __init__(self, style_image, content_image,
                                    alpha=1e4, beta=1):
        style_image - the image used as a style reference, stored as a
                        numpy.ndarray
        content_image - the image used as a content reference, stored as a
                        numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost

        -if style_image is not a np.ndarray with the shape (h, w, 3), raise a
        TypeError with the message style_image must be a numpy.ndarray with
        shape (h, w, 3)

        -if content_image is not a np.ndarray with the shape (h, w, 3), raise
        a TypeError with the message content_image must be a numpy.ndarray with
        shape (h, w, 3)

        -if alpha is not a non-negative number, raise a TypeError with the
        message alpha must be a non-negative number

        -if beta is not a non-negative number, raise a TypeError with the
        message beta must be a non-negative number

        Sets Tensorflow to execute eagerly
        Sets the instance attributes:
            style_image - the preprocessed style image
            content_image - the preprocessed content image
            alpha - the weight for content cost
            beta - the weight for style cost
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv1'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Init method for the Class

        initialize the variables
        """
        if type(style_image) is not np.ndarray or style_image.ndim != 3 \
                or style_image.shape[2] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape \
                            (h, w, 3)')
        if type(content_image) is not np.ndarray or content_image.ndim != 3 \
                or content_image.shape[2] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape \
                            (h, w, 3)')
        if type(alpha) is not int and type(alpha) is not float or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if type(beta) is not int and type(beta) is not float or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Static Method: def scale_image(image): that rescales an image such
        that its pixels values are between 0 and 1 and its largest side is 512
        pixels

        image - a numpy.ndarray of shape (h, w, 3) containing the image to be
        scaled

        -if image is not a np.ndarray with the shape (h, w, 3), raise a
        TypeError with the message image must be a numpy.ndarray with shape
        (h, w, 3)

        The scaled image should be a tf.tensor with the shape (1, h_new,
        w_new, 3) where max(h_new, w_new) == 512 and min(h_new, w_new) is
        scaled proportionately
        The image should be resized using bicubic interpolation
        After resizing, the image’s pixel values should be rescaled from the
        range [0, 255] to [0, 1].

        Returns: the scaled image
        """
        if type(image) is not np.ndarray or image.ndim != 3 \
                or image.shape[2] != 3:
            raise TypeError('image must be a numpy.ndarray with shape \
                            (h, w, 3)')
        h, w, _ = image.shape
        max_dim = 512 * (200, 100)
        maximun = max(h, w)
        scale = max_dim / maximun
        new_shape = (int(h * scale), int(w * scale))
        image = np.expand_dims(image, axis=0)
        scaled_image = tf.image.resize_bicubic(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255, 0, 1)

        return scaled_image
