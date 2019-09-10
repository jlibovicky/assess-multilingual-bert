#!/usr/bin/env python
# coding: utf-8

"""Do word alignment by minimum weighted edge cover."""

import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

from utils import vectors_for_sentence, load_bert
from mwec import edge_cover

logging.basicConfig(level=logging.INFO)


def call_bert_and_collapse_tokens(filename, model, tokenizer, layer):
    with open(filename) as f_data:
        with torch.no_grad():
            for line in f_data:
                original_tokens = line.strip().split()
                token_spans = []
                bert_tokens = []
                for token in original_tokens:
                    bert_split = tokenizer.tokenize(token)
                    token_spans.append(len(bert_split))
                    bert_tokens.extend(bert_split)

                vectors = vectors_for_sentence(
                    tokenizer, model, bert_tokens, layer,
                    skip_tokenization=True)[0].numpy()

                vectors_squeezed = []

                offset = 0
                for span in token_spans:
                    vectors_squeezed.append(
                        np.mean(vectors[offset:offset + span], axis=0))
                    offset += span

                yield np.stack(vectors_squeezed), original_tokens


def reordering_penalty(src_size, tgt_size):
    """Penalty for reordering 0-1 based on relative token distance."""
    penalty_matrix = np.zeros((src_size, tgt_size))
    max_size = max(src_size, tgt_size)
    for i in range(src_size):
        for j in range(tgt_size):
            penalty_matrix[i, j] = abs(i - j) / max_size
    return penalty_matrix


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
        "tgt", type=str, help="Sentences in the target language.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="If true, center representations first.")
    parser.add_argument(
        "--reordering-penalty", default=1e-5, type=float,
        help="Penalty for long-distance alignment added to cost.")
    parser.add_argument(
        "--verbose", default=False, action="store_true",
        help="If true, print the actual alignment.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    print(f"Loading src: {args.src}", file=sys.stderr)
    src_repr = list(call_bert_and_collapse_tokens(
        args.src, model, tokenizer, args.layer))

    print(f"Loading tgt: {args.tgt}", file=sys.stderr)
    tgt_repr = list(call_bert_and_collapse_tokens(
        args.tgt, model, tokenizer, args.layer))
    print("Data loaded.", file=sys.stderr)

    if args.center_lng:
        src_center = np.mean(
            np.concatenate([mat for mat, txt in src_repr]), 0, keepdims=True)
        tgt_center = np.mean(
            np.concatenate([mat for mat, txt in tgt_repr]), 0, keepdims=True)

        src_repr = [(mat - src_center, txt) for mat, txt in src_repr]
        tgt_repr = [(mat - tgt_center, txt) for mat, txt in tgt_repr]

    for (src_mat, src_tok), (tgt_mat, tgt_tok) in zip(src_repr, tgt_repr):
        # Cosine in (-1, 1) and MWEC requires positive weights => 2 -
        distance = 2 - (
            np.dot(src_mat, tgt_mat.T)
            / np.expand_dims(np.linalg.norm(src_mat, axis=1), 1)
            / np.expand_dims(np.linalg.norm(tgt_mat, axis=1), 0))

        if args.reordering_penalty > 0:
            distance += (
                args.reordering_penalty * reordering_penalty(*distance.shape))

        alignment = edge_cover(distance)

        if args.verbose:
            for i, token in enumerate(src_tok):
                aligned_indices = [tix for six, tix in alignment if six == i]
                aligned_formatted = [
                    f"{tgt_tok[j]} ({j})" for j in aligned_indices]
                print(f"{i:2d}: {token} -- {', '.join(aligned_formatted)}")
            print()
        else:
            print(" ".join(
                f"{src_id}-{tgt_id}" for src_id, tgt_id in alignment))


if __name__ == "__main__":
    main()
