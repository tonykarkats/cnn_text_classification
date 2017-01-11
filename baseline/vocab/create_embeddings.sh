#!/bin/bash

# This script reads the training data and creates the final word embeddings
# based on the desired percentage of the training data file.
if [ "$#" -eq 0 ]; then
    RATIO=0.01
    echo "No ratio given, using a default ratio of $RATIO."
  else
    RATIO=$1
fi

if [ ! -f vocab.txt ]; then
    echo "Building vocabulary..."
    ./build_vocab.sh
    echo "Done"
else
    echo "Vocabulary already built, skipping."
fi

if [ ! -f vocab_cut.txt ]; then
    echo "Cutting vocabulary..."
    ./cut_vocab.sh
    echo "Done"
else
    echo "Vocabulary already cut, skipping."
fi


if [ ! -f vocab.pkl ]; then
    echo "Pickling..."
    python3 pickle_vocab.py
    echo "Done"
else
    echo "Pickling already done, skipping."
fi

if [ ! -f cooc.pkl ]; then
  echo "Building co-occurrence matrix with percentage $RATIO %..."
  python3 cooc.py "$RATIO"
  echo "Done"
else
  echo "Co-occurrence matrix already built, skipping."
fi


if [ ! -f embeddings.npy ]; then
  echo "Computing glove embeddings..."
  python3 glove.py
  echo "Done"
else
  echo "Glove embeddings already computed, skipping."
fi
