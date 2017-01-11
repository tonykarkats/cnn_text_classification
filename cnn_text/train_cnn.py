#!/usr/bin/env python3

import numpy as np
import sys
import os
import time
import tensorflow as tf
from etaprogress.progress import ProgressBar

from utils import *
from tweet_cnn import *

# This is a hackish way to import the set of parameters that we want to use.
# This comes in handy to queue jobs on euler for instance.
param_set = "para_default"
if len(sys.argv[1:]) >= 1:
    param_set = sys.argv[1:][0]

exec("from " + param_set + " import *")
with open(param_set + ".py", 'r') as param_file:
    print(param_file.read())

# ==================================================


def train(x_input, y_output, vocab_size, embeddings):
    """
    Trains the CNN.
    """
    # Shuffle and split the dataset [x_input, y_output]
    shuffle_indices = np.random.permutation(np.arange(y_output.shape[0]))

    total_tweets_count = x_input.shape[0]
    print("Total # of tweets = {}".format(total_tweets_count))

    x_shuffled = x_input[shuffle_indices]
    y_shuffled = y_output[shuffle_indices]

    # k-fold Cross-Validation
    print("Training CNN with {}-fold cross-validation..".format(folds_count))

    # Initialize Saver for checkpoints
    if not os.path.exists(chckpt_folder):
        os.mkdir(chckpt_folder)
    else:
        os.system("rm -f " + chckpt_folder + "/*")

    # Maximum accuracy for each fold
    max_fold_acc = []

    def run_fold_train(fold):
        if folds_count == 1:
            # No cross validation. Use the train_split ratio
            split_point = int(len(x_shuffled) * train_split)
            x_train = x_shuffled[:split_point]
            y_train = y_shuffled[:split_point]

            x_eval = x_shuffled[split_point:]
            y_eval = y_shuffled[split_point:]
        elif folds_count == 2:
            # Use 2 folds with the given train_split
            split_point = int(len(x_shuffled) * train_split)
            print("=======\nFold {}\n=======".format(fold))

            if fold == 0:
                # First fold
                x_train = x_shuffled[:split_point]
                y_train = y_shuffled[:split_point]

                x_eval = x_shuffled[split_point:]
                y_eval = y_shuffled[split_point:]
            else:
                split_point = len(x_shuffled) - split_point
                # Second fold
                x_train = x_shuffled[split_point:]
                y_train = y_shuffled[split_point:]

                x_eval = x_shuffled[:split_point]
                y_eval = y_shuffled[:split_point]
        else:
            validation_tweets_count = int(total_tweets_count / folds_count)
            val_index_start = fold * validation_tweets_count
            val_index_end = (fold + 1) * validation_tweets_count
            print("=======\nFold {}\n=======".format(fold))

            x_eval = x_shuffled[val_index_start:val_index_end]
            y_eval = y_shuffled[val_index_start:val_index_end]

            x_train = np.vstack(
                (x_shuffled[:val_index_start], x_shuffled[val_index_end:]))
            y_train = np.vstack(
                (y_shuffled[:val_index_start], y_shuffled[val_index_end:]))

        print("Using {} tweets for training.\nUsing {} tweets for validation".format(
            len(x_train), len(x_eval)))

        # Now we have prepared the input to start training the NN
        with tf.Graph().as_default():
            tf.set_random_seed(1)
            sess = tf.Session()
            with sess.as_default():
                cnn = tweetCNN(max_tweet_length=max_tweet_length,
                               vocab_size=vocab_size,
                               initial_embeddings=embeddings,
                               embedding_dim=embedding_dim,
                               filter_sizes=filter_sizes,
                               l2_reg_lambda=l2_reg_lambda,
                               num_filters=num_filters)

                # Training Procedure
                global_step = tf.Variable(
                    0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

                sess.run(tf.initialize_all_variables())
                saver = tf.train.Saver()

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dropout_keep_prob
                    }
                    op, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy],
                        feed_dict)

                def eval_step(x_batch, y_batch):
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0  # No dropout for eval step
                    }
                    step, loss, accuracy = sess.run(
                        [global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                    return accuracy

                batches = generate_batches(
                    list(zip(x_train, y_train)), batch_size, num_epochs)
                # Evaluation batches have to be of the same size and mod has to
                # be zero
                eval_batches_count = len(x_eval) / eval_batch_size

                # Train Loop
                batches_per_epoch = len(x_train) / batch_size
                total_batches = num_epochs * batches_per_epoch
                bar_count = 0

                # Number of batches after which an evaluation step is performed
                eval_every = int(total_batches * eval_every_percent)
                print("EVAL will be performed every {} batches.".format(eval_every))

                # This counter holds the number of the last batches for which
                # the accuracy has not increased
                accuracy_not_increased_for = 0

                # Progress Bar
                bar = ProgressBar(
                    total_batches + eval_batches_count / eval_every_percent, max_width=60)

                # Number for which the accuracy hasn't increased from the best
                not_increased_for = 0
                # Maximum evaluatuon accuracy for this fold
                max_fold_eval_acc = 0.0

                for batch in batches:
                    bar_count += 1
                    bar.numerator = bar_count
                    print(bar, end='\r')
                    sys.stdout.flush()
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % eval_every == 0:
                        eval_step_id = current_step / eval_every
                        eval_batches = generate_batches(
                            list(zip(x_eval, y_eval)), eval_batch_size, 1, shuffle=False)
                        eval_acc = 0
                        for eval_batch in eval_batches:
                            bar_count += 1
                            bar.numerator = bar_count
                            print(bar, end='\r')
                            sys.stdout.flush()
                            x_eval_batch, y_eval_batch = zip(*eval_batch)
                            acc = eval_step(x_eval_batch, y_eval_batch)
                            eval_acc += acc
                        eval_acc = eval_acc / eval_batches_count
                        print("EVALUATION : Acc = {:0.5f}".format(eval_acc))

                        if eval_acc > max_fold_eval_acc:
                            not_increased_for = 0
                            max_fold_eval_acc = eval_acc
                            saver.save(sess, chckpt_folder +
                                       "/max_acc-{}.ckpt".format(fold))
                        else:
                            not_increased_for += 1
                            if not_increased_for == threshold_evals:
                                print("Evaluation accuracy hasn't increased for {} evaluation steps.. Stopping..".format(
                                    threshold_evals))
                                break
                max_fold_acc.append(max_fold_eval_acc)

    for fold in range(folds_count):
        run_fold_train(fold)

    print("===============\nFold Accuracies\n===============")
    for idx, fold_acc in enumerate(max_fold_acc):
        print("Fold {} -> {:0.5f}".format(idx, fold_acc))

    print("Average Evaluation Accuracy = {:0.5f}".format(
        sum(max_fold_acc) / folds_count))


def predict(test_input, vocab_size, embeddings):
    """
    Outputs the prediction for test_input.
    """
    sum_scores = np.zeros((len(test_input), 2))
    for fold in range(folds_count):
        with tf.Graph().as_default():
            tf.set_random_seed(1)
            sess = tf.Session()
            with sess.as_default():
                cnn = tweetCNN(max_tweet_length=max_tweet_length,
                               vocab_size=vocab_size,
                               initial_embeddings=embeddings,
                               embedding_dim=embedding_dim,
                               filter_sizes=filter_sizes,
                               l2_reg_lambda=l2_reg_lambda,
                               num_filters=num_filters)

                sess.run(tf.initialize_all_variables())
                saver = tf.train.Saver()
                saver.restore(sess, chckpt_folder +
                              "/max_acc-{}.ckpt".format(fold))
                feed_dict = {
                    cnn.input_x: test_input,
                    cnn.dropout_keep_prob: 1.0
                }
                scores = sess.run(cnn.scores, feed_dict)
                sum_scores += scores

    predictions = np.argmax(sum_scores, axis=1)
    return predictions


def write_predictions(predictions):
    """
    Write the predictions made to a file that can be submitted to Kaggle
    """
    if 'submission_file' not in globals():
        submission_file = ("../data/submission_" + param_set + "_" +
                           time.strftime("%Y-%m-%d-%H-%M") + ".csv")
    with open(submission_file, 'w') as output_file:
        output_file.write("Id,Prediction\n")
        for idx, prediction in enumerate(predictions):
            output_file.write("{},{}\n".format(
                idx + 1, -1 if prediction == 0 else 1))


def main(unit):
    np.random.seed(RANDOM_SEED)
    main_embtype = embtype
    if unit == 'char':
        pad_word = ' '
        main_embtype = ''
    else:
        pad_word = '<PAD>'
    if grouped:
        grp_suf = '_grouped'
        for f in {pos_tweets_file, neg_tweets_file, test_tweets_file}:
            split = os.path.splitext(f)
            vars()[f] = split[0] + grp_suf + split[1]

    # Loading training tweets
    print("Loading tweets...")
    tweets, y_output = load_tweets_and_outputs(
        pos_tweets_file, neg_tweets_file, unit)
    print("Done! Loaded {} tweets in total!".format(len(tweets)))

    ml = max(max_len(load_test_tweets(test_tweets_file, unit)),
             max_len(tweets))
    if ml > max_tweet_length:
        print("max length of tweets is " + str(ml))
    assert ml <= max_tweet_length

    # Pad tweets to maximum tweet length
    # TODO: FIX? Padding could be done when feeding the NN later to save memory
    #       [1,2,3] -> [1,2,3,len, len, len, len]
    print("Padding tweets to length = {}...".format(max_tweet_length))
    pad_tweets(tweets, max_tweet_length, pad_word)
    print("Done!")

    # Load vocabulary and embeddings
    # @vocab is a dictionary vocab[word] = id based on the pre-trained GloVe vocabulary.
    # @embeddings is a dictionary embeddings[id] = embedding based on the GloVe embeddings.
    # These embeddings are used to initialize the embedding table in the NN

    print("Loading vocabulary and embeddings...")
    vocab, embeddings = parse_embeddings(embedding_dim, main_embtype, pad_word)
    init_vocab_len = len(vocab)
    print("Done! Vocabulary size = {} words.".format(init_vocab_len))

    # Now that we have loaded the vocabulary for each tweet we produce a list
    # of the word indices it contains. If word embedding is unknown we initialize it
    # to random.

    print("Converting input for NN...")
    x_input = np.array(convert_input_cnn(tweets, vocab, pad_word))
    print("Done!")
    print("Done! Vocabulary size NOW = {} words.".format(len(vocab)))

    # Fill missing vocabulary embeddings
    missing_emb_count = len(vocab) - init_vocab_len
    missing_embeddings = np.zeros(
        (missing_emb_count, embedding_dim), dtype=float)
    embeddings = np.vstack((embeddings, missing_embeddings))

    train(x_input, y_output, len(vocab), embeddings)

    # =====================================================
    #                      PREDICTIONS
    # =====================================================

    print("Loading test file..")
    test_tweets = load_test_tweets(test_tweets_file, unit)
    print("Done! Loaded {} tweets in total!".format(len(test_tweets)))

    pad_tweets(test_tweets, max_tweet_length, pad_word)
    test_input = convert_test_input_cnn(test_tweets, vocab, pad_word)

    predictions = predict(test_input, len(vocab), embeddings)

    write_predictions(predictions)
    print("Predictions written!")

main('word')
