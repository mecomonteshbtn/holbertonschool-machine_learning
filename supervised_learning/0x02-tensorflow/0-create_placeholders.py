#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:51:37 2021

@author: Robinson Montes
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function create_placeholders that returns two placeholders for the NN
    Arguments:
     - nx (int): the number of feature columns in our data
     - classes (int): the number of classes in our classifier
    Returns:
    Placeholders named x and y, respectively
     * x: is the placeholder for the input data to the neural network
     * y: is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
