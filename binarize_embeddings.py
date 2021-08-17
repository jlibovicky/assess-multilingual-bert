#!/usr/bin/env python3

"""Chache binarize word embeddings table for faster loading.

The embedding loader in utils first looks if there is a cached version of the
embeddings and if yes, it uses it.
"""

import argparse

import joblib

from utils import load_word_embeddings

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("vec_file", type=str, help="File with embeddings")
    args = parser.parse_args()

    embeddings = load_word_embeddings(args.vec_file)

    joblib.dump(embeddings, args.vec_file + ".bin")


if __name__ == "__main__":
    main()
