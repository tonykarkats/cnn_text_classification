#!/usr/bin/env python3
"""
List of functions to compute the w2v embeddings
"""

import os
import word2vec as wv

# mostly taken from
# https://github.com/danielfrg/word2vec/blob/master/examples/word2vec.ipynb


def group_words(file_n):
    """
    Group up words that belong together
    """
    grp_suf = '_grouped'
    n_fn = os.path.splitext(file_n)[0] + grp_suf + os.path.splitext(file_n)[1]
    wv.word2phrase(file_n, n_fn, verbose=False)


def group_words_model():
    """
    Group the words from the full data set
    """
    group_words('tmp/train_full.txt')


def train_model(dim):
    """
    Compute the embeddings
    """
    wv.word2vec('tmp/train_full_grouped.txt',
                'model_{}.bin'.format(dim), dim, verbose=False)
