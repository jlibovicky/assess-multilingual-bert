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
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.linear_model import LinearRegression

from utils import (
    text_data_generator, batch_generator, get_repr_from_layer, load_bert)

logging.basicConfig(level=logging.INFO)


def cosine_distances(mat1, mat2):
    mat1_norms = (mat1 * mat1).sum(1, keepdim=True).sqrt()
    mat2_norms = (mat2 * mat2).sum(1).sqrt().unsqueeze(0)

    return 1 - torch.matmul(mat1, mat2.t()) / mat1_norms / mat2_norms


def euklid_distances(mat1, mat2):
    data_len = mat1.shape[0]
    differences = (
        mat1.unsqueeze(1).repeat(1, data_len, 1) -
        mat2.unsqueeze(0).repeat(data_len, 1, 1))
    return (differences ** 2).sum(2).sqrt()


def recall_at_k_from_distances(distances, k):
    """Computes recall at k using distance matrix.

    Because the data is parallel, we always want to retrieve i-th
    number from i-th row.
    """

    _, top_indices = distances.topk(k, dim=1, largest=False)
    targets = torch.arange(distances.shape[0]).unsqueeze(1)

    presence = (top_indices == targets).sum(1).float()
    return presence.mean()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
    parser.add_argument(
        "data", type=str, nargs="+",
        help="Sentences with language for training.")
    parser.add_argument(
        "--distance", choices=["cosine", "euklid"], default="cosine")
    parser.add_argument(
        "--skip-tokenization", default=False, action="store_true",
        help="Only split on spaces, skip wordpieces.")
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vector.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="Center languages to be around coordinate origin.")
    parser.add_argument(
        "--projections", default=None, nargs="+",
        help="List of sklearn projections for particular languages.")
    parser.add_argument(
        "--em-iterations", default=None, type=int,
        help="Iterations of projection self-learning.")
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
    if (args.projections is not None and
            args.em_iterations is not None):
        print("You either have pre-trained projections or self-train them.",
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    representations = []

    with torch.no_grad():
        for i, text_file in enumerate(args.data):
            print(f"Processing {text_file}")
            vectors = [
                get_repr_from_layer(
                    model, sentence_tensor, args.layer,
                    mean_pool=args.mean_pool)
                for sentence_tensor in batch_generator(
                    text_data_generator(text_file, tokenizer), 64)]

            lng_repr = torch.cat(vectors, dim=0)
            if args.center_lng:
                lng_repr = lng_repr - lng_repr.mean(0, keepdim=True)

            if projections is not None and projections[i] is not None:
                proj = projections[i]
                lng_repr = torch.from_numpy(proj.predict(lng_repr.numpy()))

            representations.append(lng_repr)

        mutual_projections = None
        if args.em_iterations is not None:
            print(f"EM training ...")
            new_mutual_projections = {}
            for i in range(args.em_iterations):
                print(f" ... iteration {i + 1}")
                for lng1, repr1 in zip(args.data, representations):
                    for lng2, repr2 in zip(args.data, representations):
                        distances = distance_fn(repr1, repr2)
                        retrieved = repr2[distances.min(dim=1)[1]]
                        proj = LinearRegression()
                        proj.fit(repr1.numpy(), retrieved.numpy())
                        new_mutual_projections[(lng1, lng2)] = proj
                mutual_projections = new_mutual_projections

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

                    if mutual_projections is not None:
                        proj = mutual_projections[(lng1, lng2)]
                        repr1 = torch.from_numpy(proj.predict(repr1.numpy()))

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
