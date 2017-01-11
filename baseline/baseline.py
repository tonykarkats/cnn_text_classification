#!/usr/bin/env python3
import pickle
import copy
import random

import numpy as np
import word2vec as wv
from scipy.sparse import *

from classifiers import *

USE_W2V_EMB = True


def get_t_embeddings_from_f(t_file, t_count, is_test, method='avg'):
    """
    Reads tweets from a file and returns the embedding
    for every tweet based on the method given.

    Tweets from training files are given one per line. (is_test = false).
    Tweets from the test file are given in the form <id>, <tweet>, and are
    processed as such when is_test == true
    """

    # Load the vocabulary
    print("Loading Vocabulary")
    with open('vocab/vocab.pkl', 'rb') as v_file:
        vocab = pickle.load(v_file)

    # Load the word embeddings
    print("Loading Embeddings")

    # Dirty: we load these embeddings even if we erase them after
    if USE_W2V_EMB:
        model = wv.load('../word2vec/model_200.bin')
        emb = model.vectors
    else:
        emb = np.load('vocab/embeddings.npy')
    embedding_dim = emb.shape[1]

    # tweet_embeddings is an np.ndarray that holds the mean embeddings per
    # tweet
    tweet_embeddings = np.empty((t_count, embedding_dim))

    with open(t_file, 'r') as tt_file:
        print("Computing mean embeddings for {}".format(t_file))

        for idx, line in enumerate(tt_file):
            # Each line is a tweet
            tweet_tokens = [vocab.get(t, -1) for t in line.strip().split()]
            if is_test:
                tweet_tokens = tweet_tokens[1:]
            tweet_tokens = [t for t in tweet_tokens if t >=
                            0 and t < emb.shape[0]]

            tweet_embedding = np.zeros(embedding_dim)
            if method == 'avg':
                # Compute average tweet embedding
                tweet_embedding = np.sum(emb[token] for token in tweet_tokens)
                if len(tweet_tokens) > 0:
                    tweet_embeddings[idx] = tweet_embedding / len(tweet_tokens)
                else:
                    # print("Tweet = {}Nothing found in vocab!".format(line))
                    tweet_embeddings[idx] = tweet_embedding
                    # All words in the tweet unknown
                if idx == t_count - 1:
                    break
            else:
                print("Unknown method : {}".format(method))
                exit()

        print("Tweet embeddings constructed! Total: {}".format(
            len(tweet_embeddings)))

        return tweet_embeddings


def shuffle_split_dataset(tweet_embeddings, training_ratio):
    """
    This functions shuffles and splits the dataset into training and validation
    sets according to the desired ratios.
    """

    validation_ratio = 1 - training_ratio

    random_seed = 0xC0FFEE
    random.seed(random_seed)
    np.random.seed(random_seed)

    t_count = len(tweet_embeddings)
    validation_count = int(validation_ratio * t_count)
    training_count = t_count - validation_count

    print("""Our dataset has {} datapoints.
            We will use {} for training and {} for validation"""
          .format(t_count, training_count, validation_count))

    to_shuffle = copy.copy(tweet_embeddings)
    random.shuffle(to_shuffle)
    return to_shuffle[:training_count], to_shuffle[training_count:]


def evaluate(predicted_tweets, target_tweets):
    """
    Returns the score
    """
    assert len(predicted_tweets) == len(target_tweets)
    tweets_count = len(target_tweets)
    return np.sum(predicted_tweets == target_tweets) / tweets_count


def save_predictions(output_file, tweet_predictions):
    """
    Saves the predictions into a submissions file.
    """
    with open(output_file, 'w') as output_f:
        output_f.write('Id,Prediction\n')
        for idx, pred in enumerate(tweet_predictions):
            output_f.write("{},{}\n".format(idx + 1, int(pred)))


def kaggle_predict(predictor, pos_train, neg_train):
    """
    Computes the predictions to submit to kaggle
    """
    test_tweets = get_t_embeddings_from_f(t_file='../data/test_data.txt',
                                          t_count=10000,
                                          is_test=True,
                                          method='avg')
    predicted = predictor(pos_train, neg_train, test_tweets)
    save_predictions("../data/submission.csv", predicted)


def main(method='log_rev', t_count=50000):
    """
    The main function, that uses one of the three methods
    """
    assert method in {'log_rev', 'lin_svc'}
    pos_tweets = get_t_embeddings_from_f(t_file='../data/train_pos_full.txt',
                                         t_count=t_count,
                                         is_test=False,
                                         method='avg')

    neg_tweets = get_t_embeddings_from_f(t_file='../data/train_neg_full.txt',
                                         t_count=t_count,
                                         is_test=False,
                                         method='avg')

    pos_train, pos_validate = shuffle_split_dataset(
        tweet_embeddings=pos_tweets, training_ratio=0.8)
    neg_train, neg_validate = shuffle_split_dataset(
        tweet_embeddings=neg_tweets, training_ratio=0.8)

    # Local run
    test_tweets = np.append(pos_validate, neg_validate, axis=0)
    if method == 'log_rev':
        predicted = predict_logistic_regression(pos_train, neg_train, test_tweets,
                                                C=10e5,
                                                tol=0.00001,
                                                max_iter=100)
    else:
        predicted = predict_linear_svc(pos_train, neg_train, test_tweets)

    target = np.ones(len(pos_validate) + len(neg_validate))
    target[len(pos_validate):] = -1
    print("Accuracy: {:0.5f}".format(evaluate(predicted, target)))

    # Uncomment for test file run
    # if method == 'log_rev':
    #     kaggle_predict(predict_logistic_regression, pos_train, neg_train)
    # elif method == 'lin_svc':
    #     kaggle_predict(predict_linear_svc, pos_train, neg_train)

if __name__ == '__main__':
    # Run the experiments for the 2 baselines
    # The SVM is more computationaly expensive so fewer datapoints are used
    main(method='log_rev', t_count=1000000)
    main(method='lin_svc', t_count=50000)
