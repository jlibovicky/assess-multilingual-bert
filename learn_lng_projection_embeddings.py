#!/usr/bin/env python

"""Learn a projection from one language space to another."""

import argparse
import logging
import sys

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

from qe_by_cosine import apply_sklearn_proj
from utils import load_word_embeddings, word_embeddings_for_file

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "data_lng1", type=str,
        help="Sentences with language for training.")
    parser.add_argument(
        "data_lng2", type=str,
        help="Sentences with language for training.")
    parser.add_argument(
        "lng1_emb", type=str, help="Source language word embeddings.")
    parser.add_argument(
        "lng2_emb", type=str, help="Target language word embeddings.")
    parser.add_argument(
        "lng1", type=str, help="Source language code.")
    parser.add_argument(
        "lng2", type=str, help="Target language code.")
    parser.add_argument(
        "save_model", type=str, help="Path to the saved model.")
    parser.add_argument(
        "--src-proj", default=None, type=str,
        help="Sklearn projection of the source language.")
    parser.add_argument(
        "--mt-proj", default=None, type=str,
        help="Sklearn projection of the target language.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.lng1} embeddings.", file=sys.stderr)
    lng1_embeddings = load_word_embeddings(args.lng1_emb)
    print(f"Loading {args.lng2} embeddings.", file=sys.stderr)
    lng2_embeddings = load_word_embeddings(args.lng2_emb)

    print(f"Loading representation for {args.data_lng1}", file=sys.stderr)
    lng1_repr = np.stack(
        word_embeddings_for_file(args.data_lng1, lng1_embeddings, args.lng1))
    print(f"Loading representation for {args.data_lng2}", file=sys.stderr)
    lng2_repr = np.stack(
        word_embeddings_for_file(args.data_lng2, lng2_embeddings, args.lng2))
    print("Representations loaded.", file=sys.stderr)

    if args.src_proj is not None:
        src_repr = apply_sklearn_proj(src_repr, args.src_proj)
    if args.mt_proj is not None:
        mt_repr = apply_sklearn_proj(mt_repr, args.mt_proj)

    print("Fitting the projection.", file=sys.stderr)
    model = LinearRegression()
    model.fit(lng1_repr, lng2_repr)
    print("Done, saving model.", file=sys.stderr)

    joblib.dump(model, args.save_model)

if __name__ == "__main__":
    main()
