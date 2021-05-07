#!/usr/bin/env python
# coding: utf-8

"""Compute centroids of BERT representations for all languages."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import (
    text_data_generator, batch_generator, get_repr_from_layer, load_bert)

import logging
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
    parser.add_argument(
        "language_list", type=str, help="TSV file with available languages.")
    parser.add_argument(
        "data", type=str, help="Directory with txt files.")
    parser.add_argument(
        "target", type=str, help="npz file with saved centroids.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vecotr.")
    parser.add_argument(
        "--batch-size", type=int, default=32)
    parser.add_argument(
        "--batch-count", type=int, default=200)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    language_names = []
    centroids = []

    with open(args.language_list) as lng_f:
        for line in lng_f:
            name, code = line.strip().split("\t")
            data_file = os.path.join(args.data, f"{code}.txt")

            data = text_data_generator(data_file, tokenizer)
            batches = batch_generator(data, args.batch_size, tokenizer)
            print(f"Data iterator initialized: {data_file}")

            with torch.no_grad():
                representations = []
                for _, txt in zip(range(args.batch_count), batches):
                    batch_repr = get_repr_from_layer(
                        model, txt.to(device), args.layer,
                        tokenizer.pad_token_id,
                        mean_pool=args.mean_pool).cpu().numpy()
                    if not np.any(np.isnan(batch_repr)):
                        representations.append(batch_repr)

                if representations:
                    language_names.append(name)
                    centroid = np.concatenate(representations, axis=0).mean(0)
                    centroids.append(centroid)

    print("Centroids computed.")

    np.savez(args.target, languages=language_names, centroids=centroids)


if __name__ == "__main__":
    main()
