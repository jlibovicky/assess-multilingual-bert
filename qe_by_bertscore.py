#!/usr/bin/env python
# coding: utf-8

"""Compute centroids of BERT representations for all languages."""

import argparse
import logging
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import vectors_for_sentence, load_bert

logging.basicConfig(level=logging.INFO)


def apply_sklearn_proj(representations, model_path):
    print("Projecting representations.", file=sys.stderr)
    model = joblib.load(model_path)
    return [model.predict(rep) for rep in representations]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
    parser.add_argument(
        "src", type=str, help="Sentences in source language.")
    parser.add_argument(
        "mt", type=str, help="Machine-translated sentences.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="If true, center representations first.")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    print(f"Loading src: {args.src}", file=sys.stderr)
    with open(args.src) as f_src:
        with torch.no_grad():
            src_repr = [
                vectors_for_sentence(
                    tokenizer, model, line.rstrip(), args.layer)[0].numpy()
                for line in f_src]

    print(f"Loading mt: {args.mt}", file=sys.stderr)
    with open(args.mt) as f_mt:
        with torch.no_grad():
            mt_repr = [
                vectors_for_sentence(
                    tokenizer, model, line.rstrip(), args.layer)[0].numpy()
                for line in f_mt]

    if args.center_lng:
        src_center = np.mean(np.concatenate(src_repr), 0)
        mt_center = np.mean(np.concatenate(mt_repr), 0)

        src_repr = [r - src_center for r in src_repr]
        mt_repr = [r - mt_center for r in mt_repr]

    if args.src_proj is not None:
        src_repr = apply_sklearn_proj(src_repr, args.src_proj)
    if args.mt_proj is not None:
        mt_repr = apply_sklearn_proj(mt_repr, args.mt_proj)

    for src, mt in zip(src_repr, mt_repr):
        similarity = (
            np.dot(src, mt.T)
            / np.expand_dims(np.linalg.norm(src, axis=1), 1)
            / np.expand_dims(np.linalg.norm(mt, axis=1), 0))

        recall = similarity.max(1).sum() / similarity.shape[0]
        precision = similarity.max(0).sum() / similarity.shape[1]

        if recall + precision > 0:
            f_score = 2 * recall * precision / (recall + precision)
        else:
            f_score = 0

        print(f_score)


if __name__ == "__main__":
    main()
