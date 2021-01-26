#!/usr/bin/env python3

import tensorflow as tf

build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    model = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    optimize_model(model, 0.01, 0.99, 0.9)
    print(model.loss)
    print(model.metrics)
    opt = model.optimizer
    print(opt.__class__)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run((opt.lr, opt.beta_1, opt.beta_2))) 
