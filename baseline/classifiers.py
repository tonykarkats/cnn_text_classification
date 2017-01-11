#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from scipy.sparse import *
from sklearn import linear_model, svm

import random

"""
Prepares the datasets for the classification operations
"""


def prepare_datasets(pos_tweet_embeddings, neg_tweet_embeddings):
    """Returns the tweet embeddings and their classification
    """

    pos_tweet_count = len(pos_tweet_embeddings)
    neg_tweet_count = len(neg_tweet_embeddings)
    total_tweet_count = pos_tweet_count + neg_tweet_count

    tweet_embeddings = np.append(
        pos_tweet_embeddings, neg_tweet_embeddings, axis=0)

    print("Training logistic regression classifier with {} tweet embeddings"
          .format(total_tweet_count))
    # Outputs for :) = 1, -1 for :(
    outputs = np.ones(pos_tweet_count + neg_tweet_count)
    outputs[pos_tweet_count:] = -1

    return tweet_embeddings, outputs


def predict_logistic_regression(pos_tweet_embeddings,
                                neg_tweet_embeddings,
                                test_embeddings,
                                C=10e5,
                                tol=0.00001,
                                max_iter=100):
    """The three last arguments are:
    _the inverse regularization strength
    _the stopping criteria tolerance
    _the maximum number of iterations taken for the solver to converge
    """
    tweet_embeddings, outputs = prepare_datasets(
        pos_tweet_embeddings, neg_tweet_embeddings)
    total_tweet_count = len(tweet_embeddings)

    # Train classifier: Logistic Regression
    print("Training logistic regression classifier with {} tweet embeddings"
          .format(total_tweet_count))
    logistic = linear_model.LogisticRegression(C=C, tol=tol, max_iter=max_iter)
    logistic.fit(tweet_embeddings, outputs)

    # Predict on test embeddings
    test_count = len(test_embeddings)
    print("Predicting for {} test tweets...".format(test_count))
    predicted = logistic.predict(test_embeddings)
    print("Done")

    return predicted


def predict_linear_svc(pos_tweet_embeddings,
                       neg_tweet_embeddings, test_embeddings):
    """Uses svm.svc
    """
    pos_tweet_count = len(pos_tweet_embeddings)
    neg_tweet_count = len(neg_tweet_embeddings)
    total_tweet_count = pos_tweet_count + neg_tweet_count

    tweet_embeddings = np.append(
        pos_tweet_embeddings, neg_tweet_embeddings, axis=0)

    print("Training linearSVC classifier with {} tweet embeddings"
          .format(total_tweet_count))
    # Outputs for :) = 1, -1 for :(
    outputs = np.ones(pos_tweet_count + neg_tweet_count)
    outputs[pos_tweet_count:] = -1

    # Train classifier : Linear SVM
    clf = svm.SVC()
    clf.fit(tweet_embeddings, outputs)

    # Predict on test embeddings
    test_count = len(test_embeddings)
    print("Predicting for {} test tweets...".format(test_count))
    predicted = clf.predict(test_embeddings)
    print("Done")

    return predicted
