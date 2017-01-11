#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

# This script performs the following actions:
# 1) Concatenates the positive and negative training data 
# 2) Make every token (delimited by space) a separate line and remove blank lines
# 3) *NEW* Remove hashtags
# 4) Sorts the words
# 5) Removes duplicates annotating every word with the number of occurrences

cat ../../data/train_pos_full.txt ../../data/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sed "s/#//g" | sort | uniq -c > vocab.txt
