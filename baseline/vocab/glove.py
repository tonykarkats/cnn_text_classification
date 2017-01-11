#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def compute_glove_embeddings(embedding_dim, nmax, alpha, eta, epochs):

    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save('embeddings', xs)


if __name__ == '__main__':
    compute_glove_embeddings(embedding_dim = 20,
                             nmax = 100,
                             alpha = 3/4,
                             eta = 0.001,
                             epochs = 10)
