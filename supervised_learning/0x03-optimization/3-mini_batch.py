#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def get_batch(t, batch_size):
    """
    Helper function to divide data in batches
    """

    batch_list = []
    i = 0
    m = t.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    for b in range(batches):
        if b != batches - 1:
            batch_list.append(t[i:(i + batch_size)])
        else:
            batch_list.append(t[i:])
        i += batch_size

    return batch_list


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Function that trains a loaded neural network model using mini-batch
    gradient descent
    Arguments:
     - X_train: is a numpy.ndarray of shape (m, 784) containing the
                training data
        * m is the number of data points
        * 784 is the number of input features
     - Y_train: is a one-hot numpy.ndarray of shape (m, 10) containing
                the training labels
        * 10 is the number of classes the model should classify
     - X_valid: is a numpy.ndarray of shape (m, 784) containing the
                validation data
     - Y_valid: is a one-hot numpy.ndarray of shape (m, 10) containing
                the validation labels
     - batch_size: is the number of data points in a batch
     - epochs: is the number of times the training should pass through
               the whole dataset
     - load_path: is the path from which to load the model
     - save_path: is the path to where the model should be saved after training
    Returns:
    The path where the model was saved
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
            X, Y = shuffle_data(X_train, Y_train)
            loss_t, acc_t = sess.run((loss, accuracy),
                                     feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if i < epochs:
                X_batch = get_batch(X, batch_size)
                Y_bathc = get_batch(Y, batch_size)
                for b in range(1, len(X_batch) + 1):
                    sess.run(train_op, feed_dict={x: X_batch[b - 1],
                                                  y: Y_bathc[b - 1]})

                    loss_t, acc_t = sess.run((loss, accuracy),
                                             feed_dict={x: X_batch[b - 1],
                                                        y: Y_bathc[b - 1]})

                    if not b % 100:
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(loss_t))
                        print('\t\tAccuracy: {}'.format(acc_t))

        save_path = saver.save(sess, save_path)

    return save_path
