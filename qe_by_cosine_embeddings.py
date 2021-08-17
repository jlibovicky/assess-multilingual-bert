#!/usr/bin/env python
# coding: utf-8

"""Compute centroids of BERT representations for all languages."""

import argparse
import logging
import sys

import numpy as np
import torch
from qe_by_cosine import apply_sklearn_proj
from utils import load_word_embeddings, word_embeddings_for_file

logging.basicConfig(level=logging.INFO)


def center(lng_repr):
    return lng_repr - lng_repr.mean(0, keepdim=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "src", type=str, help="Sentences in source language.")
    parser.add_argument(
        "mt", type=str, help="Sentences in the target language.")
    parser.add_argument(
        "src_emb", type=str, help="Source language word embeddings.")
    parser.add_argument(
        "mt_emb", type=str, help="Target language word embeddings.")
    parser.add_argument(
        "src_lng", type=str, help="Source language code.")
    parser.add_argument(
        "mt_lng", type=str, help="Target language code.")
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vecotr.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="If true, center representations first.")
    parser.add_argument(
        "--batch-size", type=int, default=32)
    parser.add_argument(
        "--src-proj", default=None, type=str,
        help="Sklearn projection of the source language.")
    parser.add_argument(
        "--mt-proj", default=None, type=str,
        help="Sklearn projection of the target language.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.center_lng and (
            args.src_proj is not None and args.src_proj is not None):
        print("You can either project or center "
              "the representations, not both.", file=sys.stderr)
        exit(1)

    torch.set_num_threads(args.num_threads)

    src_embeddings = load_word_embeddings(args.src_emb)
    mt_embeddings = load_word_embeddings(args.mt_emb)

    src_repr = torch.from_numpy(np.stack(
        word_embeddings_for_file(args.src, src_embeddings, args.src_lng)))
    mt_repr = torch.from_numpy(np.stack(
        word_embeddings_for_file(args.mt, mt_embeddings, args.mt_lng)))

    if args.center_lng:
        src_repr = center(src_repr)
        mt_repr = center(mt_repr)

    if args.src_proj is not None:
        src_repr = apply_sklearn_proj(src_repr, args.src_proj)
    if args.mt_proj is not None:
        mt_repr = apply_sklearn_proj(mt_repr, args.mt_proj)


    src_norm = (src_repr * src_repr).sum(1).sqrt()
    mt_norm = (mt_repr * mt_repr).sum(1).sqrt()

    cosine = (src_repr * mt_repr).sum(1) / src_norm / mt_norm

    for num in cosine.cpu().detach().numpy():
        print(num)


if __name__ == "__main__":
    main()
