# ===========================================
# PARAMETERS
# ============================================

embedding_dim = 45
# k-fold cross validation
folds_count = 1
train_split = 0.95
eval_every_percent = 0.01  # Frequency of evaluation
batch_size = 1024  # Parameters for input
eval_batch_size = 100
num_epochs = 2  # feed.
filter_sizes = [6,7,8]
num_filters = 100
learning_rate = 1e-3
RANDOM_SEED = 0xC0FFEE
l2_reg_lambda = 1.0
max_tweet_length = 450
dropout_keep_prob = 0.5
embtype = 'wv'  # either 'wv' or 'glove'
# If the accuracy does not increase for these many evaluation batches stop
threshold_evals = 10
pos_tweets_file = '../data/train_pos_full.txt'
neg_tweets_file = '../data/train_neg_full.txt'
test_tweets_file = '../data/test_data.txt'
# submission_file = '../data/submission.csv'
chckpt_folder = "./.checkpoints"
grouped = True  # use the grouping provided by word2vec
