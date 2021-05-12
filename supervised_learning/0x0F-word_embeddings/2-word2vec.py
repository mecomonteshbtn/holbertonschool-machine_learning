#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:12:42 2021

@author: Robinson Montes
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    Function that creates and trains a gensim word2vec model

    Arguments:
     - sentences is a list of sentences to be trained on
     - size is the dimensionality of the embedding layer
     - min_count is the minimum number of occurrences of a word
        for use in training
     - window is the maximum distance between the current and
        predicted word within a sentence
     - negative is the size of negative sampling
     - cbow is a boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
     - iterations is the number of iterations to train over
     - seed is the seed for the random number generator
     - workers is the number of worker threads to train the model

    Returns:
     The trained model
    """

    model = Word2Vec(sentences=sentences, min_count=min_count,
                     iter=iterations, size=size,
                     window=window, sg=cbow,
                     seed=seed, negative=negative)
    model.train(sentences=sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
