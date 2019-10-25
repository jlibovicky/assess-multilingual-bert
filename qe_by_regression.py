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
import torch.nn as nn
import torch.optim as optim

from utils import (
    text_data_generator, batch_generator, load_bert)
from qe_by_cosine import repr_for_txt_file

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    if not args.exclude_src:
        train_src_repr = repr_for_txt_file(
            args.train_src, tokenizer, model, device, args.layer,
            center_lng=False, mean_pool=args.mean_pool).numpy()
    if not args.exclude_mt:
        train_mt_repr = repr_for_txt_file(
            args.train_mt, tokenizer, model, device, args.layer,
            center_lng=False, mean_pool=args.mean_pool).numpy()
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
        test_src_repr = repr_for_txt_file(
            args.test_src, tokenizer, model, device, args.layer,
            center_lng=False, mean_pool=args.mean_pool).numpy()
    if not args.exclude_mt:
        test_mt_repr = repr_for_txt_file(
            args.test_mt, tokenizer, model, device, args.layer,
            center_lng=False, mean_pool=args.mean_pool).numpy()

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
