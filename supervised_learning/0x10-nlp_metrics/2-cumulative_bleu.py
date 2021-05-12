#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:12:42 2021

@author: Robinson Montes
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Function that calculates the cumulative n-gram BLEU score for a sentence

    Arguments:
     - references is a list of reference translations
        * each reference translation is a list of the words in the translation
     - sentence is a list containing the model proposed sentence
     - n is the size of the largest n-gram to use for evaluation

    Returns:
     The cumulative n-gram BLEU score
    """

    len_refer = []
    clipped = {}
    len_sentence = len(sentence)

    N_sentence = [' '.join([str(jd) for jd in sentence[id:id + n]])
                  for id in range(len(sentence) - (n - 1))]
    len_Noutput = (len(N_sentence))

    for refs in references:
        N_reference = [' '.join([str(jd) for jd in refs[id:id + n]])
                       for id in range(len(sentence) - (n - 1))]

        len_refer.append(len(refs))

        for w in N_reference:
            if w in N_sentence:
                if not clipped.keys() == w:
                    clipped[w] = 1

    clipped_count = sum(clipped.values())
    closest_idx = np.argmin([abs(len(x) - len_Noutput) for x in references])
    closest_len_refer = len(references[closest_idx])

    if len_sentence > closest_len_refer:
        bp = 1
    else:
        bp = np.exp(1 - (float(closest_len_refer) / len_sentence))

    nFrac = np.empty((n,))
    nFrac[:] = 1 / n
    bleu = clipped_count / len_Noutput
    BLEU_score = bp * np.exp(np.sum(nFrac * np.log(clipped_count /
                                                   len_Noutput)))

    return BLEU_score
