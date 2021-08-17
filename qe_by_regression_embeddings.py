#!/usr/bin/env python
# coding: utf-8

"""Compute centroids of BERT representations for all languages."""

import argparse
import os
import logging
import sys

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
import torch

from utils import load_word_embeddings, word_embeddings_for_file

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "train_src", type=str,
        help="Sentences in source language for training.")
    parser.add_argument(
        "train_mt", type=str,
        help="Machine-translated sentences for training.")
    parser.add_argument(
        "train_hter", type=str,
        help="Machine-translated sentences for training.")
    parser.add_argument(
        "test_src", type=str, help="Sentences in source language for testing.")
    parser.add_argument(
        "test_mt", type=str, help="Machine-translated sentences for testing.")
    parser.add_argument(
        "src_emb", type=str, help="Source language word embeddings.")
    parser.add_argument(
        "mt_emb", type=str, help="Target language word embeddings.")
    parser.add_argument(
        "src_lng", type=str, help="Source language code.")
    parser.add_argument(
        "mt_lng", type=str, help="Target language code.")
    parser.add_argument(
        "--exclude-src", default=False, action="store_true",
        help="Exclude source representatiion from the classifier.")
    parser.add_argument(
        "--exclude-mt", default=False, action="store_true",
        help="Exclude target representatiion from the classifier.")
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vecotr.")
    parser.add_argument(
        "--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.exclude_src and args.exclude_mt:
        print("You cannot exclude both source and MT!", file=sys.stderr)
        exit(1)

    torch.set_num_threads(args.num_threads)

    src_embeddings = load_word_embeddings(args.src_emb)
    mt_embeddings = load_word_embeddings(args.mt_emb)

    if not args.exclude_src:
        train_src_repr = word_embeddings_for_file(
            args.train_src, src_embeddings, args.src_lng)
    if not args.exclude_mt:
        train_mt_repr = word_embeddings_for_file(
            args.train_src, mt_embeddings, args.mt_lng)

    if args.exclude_src:
        train_inputs = train_mt_repr
    elif args.exclude_mt:
        train_inputs = train_src_repr
    else:
        train_inputs = np.concatenate((train_src_repr, train_mt_repr), axis=1)

    with open(args.train_hter) as f_tgt:
        train_targets = np.array([float(line.rstrip()) for line in f_tgt])

    print("Training regression ... ", file=sys.stderr, end="", flush=True)
    regressor = MLPRegressor((256), early_stopping=True)
    regressor.fit(train_inputs, train_targets)
    print("Done.", file=sys.stderr)

    if not args.exclude_src:
        test_src_repr = word_embeddings_for_file(
            args.test_src, src_embeddings, args.src_lng)
    if not args.exclude_mt:
        test_mt_repr = word_embeddings_for_file(
            args.test_src, mt_embeddings, args.mt_lng)

    if args.exclude_src:
        test_inputs = test_mt_repr
    elif args.exclude_mt:
        test_inputs = test_src_repr
    else:
        test_inputs = np.concatenate((test_src_repr, test_mt_repr), axis=1)

    predictions = regressor.predict(test_inputs)

    for num in predictions:
        print(num)


if __name__ == "__main__":
    main()
