#!/usr/bin/env python3
import pickle


"""
This script serializes the word vocabulary usign Python Pickle and exports it
as 'vocab.pkl'.
"""


def main():
    vocab = dict()
    with open('vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
