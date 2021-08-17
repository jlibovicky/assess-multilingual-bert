#!/usr/bin/env python

"""Probe multilingual BERT on cross-lingual retrieval."""

import argparse
import logging
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

from sentence_retrieval import (
    cosine_distances, euklid_distances, recall_at_k_from_distances)
from utils import (
    load_word_embeddings, word_embeddings_for_file)

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "data", type=str, nargs="+",
        help="Sentences with language for training.")
    parser.add_argument(
        "--embeddings", type=str, nargs="+",
        help="Files with word embeddings.")
    parser.add_argument(
        "--languages", type=str, nargs="+",
        help="Language codes used for tokenizers.")
    parser.add_argument(
        "--distance", choices=["cosine", "euklid"], default="cosine")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="Center languages to be around coordinate origin.")
    parser.add_argument(
        "--projections", default=None, nargs="+",
        help="List of sklearn projections for particular languages.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.center_lng and args.projections is not None:
        print("You cannot do projections and centering at once.",
              file=sys.stderr)
        exit(1)
    if (args.projections is not None and
            len(args.projections) != len(args.data)):
        print("You must have a projection for each data file.",
              file=sys.stderr)
        exit(1)

    projections = None
    if args.projections is not None:
        projections = []
        for proj_str in args.projections:
            if proj_str == "None":
                projections.append(None)
            else:
                projections.append(joblib.load(proj_str))

    distance_fn = None
    if args.distance == "cosine":
        distance_fn = cosine_distances
    elif args.distance == "euklid":
        distance_fn = euklid_distances
    else:
        raise ValueError("Unknown distance function.")

    torch.set_num_threads(args.num_threads)

    representations = []

    with torch.no_grad():
        for i, (text_file, embeddings_file, lng) in enumerate(
                zip(args.data, args.embeddings, args.languages)):
            print(f"Loading embeddings {embeddings_file}")
            word_embeddings = load_word_embeddings(embeddings_file)
            print(f"Processing {text_file}")
            embedded_sentences = word_embeddings_for_file(
                text_file, word_embeddings, lng)

            lng_repr = torch.from_numpy(np.stack(embedded_sentences))
            if args.center_lng:
                lng_repr = lng_repr - lng_repr.mean(0, keepdim=True)

            if projections is not None and projections[i] is not None:
                proj = projections[i]
                lng_repr = torch.from_numpy(proj.predict(lng_repr.numpy()))

            representations.append(lng_repr)

        data_len = representations[0].shape[0]
        assert all(r.shape[0] == data_len for r in representations)
        print()
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"Recall at {k}, random baseline {k / data_len:.5f}")
            print("--", end="\t")
            for lng in args.data:
                print(lng[-6:-4], end="\t")
            print()

            recalls_to_avg = []

            for lng1, repr1 in zip(args.data, representations):
                print(lng1[-6:-4], end="\t")
                for lng2, repr2 in zip(args.data, representations):

                    distances = distance_fn(repr1, repr2)

                    recall = recall_at_k_from_distances(distances, k)
                    print(f"{recall.numpy():.5f}", end="\t")

                    if lng1 != lng2:
                        recalls_to_avg.append(recall.numpy())
                print()
            print(f"On average: {np.mean(recalls_to_avg):.5f}")
            print()


if __name__ == "__main__":
    main()
