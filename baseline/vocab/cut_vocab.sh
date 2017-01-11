#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

# This script:
# 1) Removes spaces in the beggining of each line created by sed from the previous script.
# 2) Reverse sorts according to numerical value (most frequent -> less frequent)
# 3) Removes all the words that have <= 4 occurrences
# 4) Removes the numerical occurrences values.
#

sed "s/^\s\+//g" < vocab.txt | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt
