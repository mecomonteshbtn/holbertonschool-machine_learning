#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:32:21 2021

@author: Robinson Montes
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Class that loads and preps a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        Instance attributes:
         - data_train, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervided
         - data_valid, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervided
         - tokenizer_pt is the Portuguese tokenizer created
            from the training set
         - tokenizer_en is the English tokenizer created
            from the training set
        """

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Instance method that creates sub-word tokenizers for our dataset

        Arguments:
         - data is a tf.data.Dataset whose examples are
            formatted as a tuple (pt, en)
            * pt is the tf.Tensor containing the Portuguese sentence
            * en is the tf.Tensor containing the corresponding English sentence

        Returns:
         tokenizer_pt, tokenizer_en
          - tokenizer_pt is the Portuguese tokenizer
          - tokenizer_en is the English tokenizer
        """

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2**15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        instance method that encodes a translation into tokens

        Arguments:
        - pt is the tf.Tensor containing the Portuguese sentence
        - en is the tf.Tensor containing the corresponding English sentence

        Returns:
        pt_tokens, en_tokens
        - pt_tokens is a tf.Tensor containing the Portuguese tokens
        - en_tokens is a tf.Tensor containing the English tokens
        """

        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens
