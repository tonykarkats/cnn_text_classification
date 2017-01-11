import tensorflow as tf
import numpy as np


class tweetCNN(object):
    """
    A Convolutional Neural Network for tweet sentiment analysis.
    The architecture is implemented as described by the paper:
    "Convolutional Neural Networks for Sentence Classification" from Kim et al.

    Parts of the implementation are taken from the blog:
    "Implementing a CNN for text classification in Tensorflow"

    @ max_tweet_length   : The maximum tweet length (usually 140).
    @ vocab_size         : The size of the vocabulary.
    @ initial_embeddings : Initialization of embeddings matrix.
                           For every word in the vocabulary contains the
                           pre-trained embeddings. -> Shape : [words,
                           embedding_dim]
    @ filter_sizes       : Filter sizes to be used. The filter size is
                           equivalent to the "window" of words the filter will
                           go over every time.
    @ l2_reg_lambda      : The regularization parameter to be used.
    @ num_filters        : The number of filters to be used for every filter
                           size.

    * Note that the total number of filters that will be used is:
      num_filters * len(filter_sizes)
    """

    def __init__(
            self, max_tweet_length, vocab_size, initial_embeddings,
            embedding_dim, filter_sizes, l2_reg_lambda, num_filters):

        # Placeholders
        # ============
        # Placeholder for input [num_batches, max_tweet_length]
        self.input_x = tf.placeholder(
            tf.int32, [None, max_tweet_length], name="input_x")
        # Placeholder for output [num_batches, 2].  2 is for the 2 classes.
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")

        # Placeholder for dropout probability to avoid overfitting.
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.convert_to_tensor(
                initial_embeddings, dtype=tf.float32), name="W")
            self.embedding = tf.nn.embedding_lookup(W, self.input_x)
            # The input to the conv2D must be 4D [batch, max_tweet_length,
            # embedding_dim, -1]
            # -1 is dummy (channel)
            self.embedding_with_channel = tf.expand_dims(self.embedding, -1)

        # Convolution and Max-Pooling Layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedding_with_channel,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply ReLu to the convolution output
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_tweet_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(
                [num_filters_total, 2], stddev=0.1, name="W"))
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
