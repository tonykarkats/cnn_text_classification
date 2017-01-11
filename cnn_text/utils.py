"""Various helpers
"""
import re
from multiprocessing import Pool
import word2vec as wv
import numpy as np
import operator


def clean_str(string):
    """
    String cleaning.
    From: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    >>> clean_str("I'll clean this (string)")
    "i 'll clean this ( string )"
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r"#", "", string)
    return string.strip().lower()


def clean_tweets(tweets):
    """
    Applies clean_str to all the tweets.
    """
    # with Pool() as pool:
    #     tweets = pool.map(clean_str, tweets)
    clean = [clean_str(tweet) for tweet in tweets]

    return clean


def split_tweet(tweet, unit):
    """
    Splits the tweet according to the chosen unit, e.g. per word.
    """
    assert unit in {'word', 'char'}
    if unit == 'word':
        return tweet.split(' ')
    elif unit == 'char':
        return list(tweet)


def preprocess(tweets, unit):
    """
    Applies the preprocessing on the tweets.
    """
    if unit == 'word':
        tweets = clean_tweets(tweets)
    return [split_tweet(s, unit) for s in tweets]


def load_tweets(tweets_file, unit='word'):
    """
    Loads and preprocess tweets from a data file.
    """
    tweets = list(
        open(tweets_file, "r", encoding='utf-8').readlines())
    return preprocess(tweets, unit)


def load_test_tweets(test_tweets_file, unit):
    """
    Loads the test tweets.
    """
    test_tweets = list(
        open(test_tweets_file, "r", encoding='utf-8').readlines())
    test_tweets = [str(s.split(',')[1:]) for s in test_tweets]
    return preprocess(test_tweets, unit)


def max_len(tweets):
    """
    Returns the length of the longest tweet
    """
    return len(max(tweets, key=len))


def load_tweets_and_outputs(pos_tweets_file, neg_tweets_file, unit):
    """
    Loads the positive and negative Tweets and appends the respective labels.

    Returns tweets : A 2D matrix containing the tweets split by words
    and the related classification.
    """
    # Load data from files
    positive_tweets = load_tweets(pos_tweets_file, unit)
    negative_tweets = load_tweets(neg_tweets_file, unit)

    # Split tweets by words
    tweets = positive_tweets + negative_tweets

    positive_outputs = [[0, 1] for _ in positive_tweets]
    negative_outputs = [[1, 0] for _ in negative_tweets]
    output = np.array(np.concatenate([positive_outputs, negative_outputs], 0))

    return [tweets, output]


def pad_tweets(tweets, max_tweet_length, pad_word):
    """
    Pads tweets to the specified length with the pad_word
    """
    for i, _ in enumerate(tweets):
        padding_chars = max_tweet_length - len(tweets[i])
        assert 0 <= padding_chars
        tweets[i].extend(padding_chars * [pad_word])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for _ in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def parse_glove_embeddings(embedding_dim):
    """
    Parses the glove pretrained embeddings and returns two dictionaries:
    @vocab is a dictionary vocab[word] = id based on the pre-trained GloVe
    vocabulary.
    @embeddings is a dictionary embeddings[id] = embedding based on the GloVe
    embeddings.

    @embedding_dim is the number of features to keep. We can only choose
    between {25,50,100,200} as these are the available ones from the glove
    datasets.
    """
    if embedding_dim not in [25, 50, 100, 200]:
        print("Error. Embedding dimension not in {25, 50, 100, 200}")
        return

    input_file = '../data/glove.twitter.27B.{}d.txt'.format(embedding_dim)
    print("Parsing {}. Keeping {} features for each word..".format(
        input_file, embedding_dim))
    embeddings = []
    vocab = dict()
    with open(input_file, 'r', encoding='utf-8') as infile:
        error_count = 0
        for word_id, line in enumerate(infile):
            line_tokens = line.split()
            word = line_tokens[0]
            embedding = [float(x) for x in line_tokens[1:]]
            if len(embedding) != embedding_dim:
                print("Error parsing one embedding given")
                print("ID: " + word_id.__str__())
                print("Skipping this embedding...")
                error_count = error_count + 1
            else:
                embeddings.append(embedding)
                vocab[word] = word_id
        print("Parsing of input_file done, " +
              error_count.__str__() + " errors")
    return vocab, embeddings


def parse_wv_embeddings(embedding_dim):
    """Parse the word2vec embeddings for the given dimension

    You need to create the w2v embeddings beforehand, using the script.
    """
    vocab = dict()
    try:
        model = wv.load('../word2vec/model_{}.bin'.format(embedding_dim))
    except FileNotFoundError:
        print("""Model not found in the word2vec folder,
                 you need to compute the model for the given dimension.""")
        exit(1)
    assert model.vectors.shape[1] == embedding_dim
    for ind in range(len(model.vocab)):
        vocab[model.vocab[ind]] = ind
    return vocab, model.vectors


def compute_char_embeddings(embedding_dim):
    """
    Computes null char embeddings
    """
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                 'y', 'z', ' ', '!', '\'', '(', ')', ',', '0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9', '?', '`', '_']
    # assertion to make sure the one-hot embeddings make sense
    assert len(char_list) <= embedding_dim
    print("char_list is " + str(len(char_list)) + " characters long.")
    vocab = dict()
    embeddings = np.eye(len(char_list), embedding_dim)
    for ind, char in enumerate(char_list):
        vocab[char] = ind
    return vocab, embeddings


def parse_embeddings(embedding_dim, embtype, pad_word):
    """Parse embeddings, using either glove or w2v embeddings
    """
    if embtype == 'glove':
        vocab, embeddings = parse_glove_embeddings(embedding_dim)
    elif embtype == 'wv':
        vocab, embeddings = parse_wv_embeddings(embedding_dim)
    else:  # no pre-computed embeddings
        vocab, embeddings = compute_char_embeddings(embedding_dim)

    # Add special pad_word
    pad_id = len(vocab)
    vocab[pad_word] = pad_id

    embeddings = np.array(embeddings, dtype=float)
    embeddings = np.vstack((embeddings, np.zeros(embedding_dim, dtype=float)))

    print("Done!")
    return vocab, embeddings


def generate_batches(data, batch_size, num_epochs, shuffle=True):
    """
    Generates batches for the NN input feed.

    Returns a generator (yield) as the datasets are expected to be huge.
    """
    data = np.array(data)
    data_size = len(data)
    batches_per_epoch = int(data_size / batch_size)
    print("Generating batches.. Total # of batches = {}".format(
        batches_per_epoch * num_epochs))
    for _ in range(num_epochs):
        if shuffle:
            shuffle_ind = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_ind]
        else:
            shuffled_data = data
        for batch_num in range(batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def convert_input_cnn(tweets, vocab, pad_word):
    """
    Converts the tweets to a suitable form for the CNN
    If the word is not in the vocab, we substitute it with pad_word.
    """
    total_words_count = 0
    unknown_words_count = 0
    unknown_words = dict()
    x_input = []
    for tweet in tweets:
        words_list = []
        for word in tweet:
            if word != '<PAD>':
                total_words_count += 1
            if word not in vocab:
                unknown_words_count += 1
                if word not in unknown_words:
                    unknown_words[word] = 1
                else:
                    unknown_words[word] += 1
                vocab[word] = len(vocab)
            words_list.append(vocab[word])
        x_input.append(words_list)
    print("Unknown words = {}".format(unknown_words_count))
    return x_input


def convert_test_input_cnn(tweets, vocab, pad_word):
    """
    Converts the tweets to a suitable form for the CNN.
    Use only for test input. Does not add new words to vocab
    """
    total_words_count = 0
    unknown_words_count = 0
    unknown_words = dict()
    x_input = []
    for tweet in tweets:
        words_list = []
        for word in tweet:
            if word != '<PAD>':
                total_words_count += 1
            if word not in vocab:
                unknown_words_count += 1
                if word not in unknown_words:
                    unknown_words[word] = 1
                else:
                    unknown_words[word] += 1
                word = pad_word
            words_list.append(vocab[word])
        x_input.append(words_list)
    print("Unknown words = {}".format(unknown_words_count))
    return x_input


if __name__ == "__main__":
    import doctest
    doctest.testmod()
