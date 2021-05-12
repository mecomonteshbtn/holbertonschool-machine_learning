#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:12:42 2021

@author: Robinson Montes
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram BLEU score for a sentence

    Arguments:
     - references is a list of reference translations
        * each reference translation is a list of the words in the translation
     - sentence is a list containing the model proposed sentence

    Returns:
     The unigram BLEU score
    """

    len_output = len(sentence)
    len_refer = []
    clipped = {}

    # References and sentences
    for ref in references:
        len_refer.append(len(ref))

        for w in ref:
            if w in sentence:
                if not clipped.keys() == w:
                    clipped[w] = 1

    clipped_count = sum(clipped.values())
    closest_idx = np.argmin([abs(len(x) - len_output) for x in references])
    closest_len_refer = len(references[closest_idx])

    if len_output > closest_len_refer:
        bp = 1
    else:
        bp = np.exp(1 - float(closest_len_refer) / float(len_output))

    # BLEU_score = bp * (clipped_count / len_output)
    BLEU_score = bp * np.exp(np.log(clipped_count / len_output))

    return BLEU_score
