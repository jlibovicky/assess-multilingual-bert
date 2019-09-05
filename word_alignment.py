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
                vectors_torch, tokenized = vectors_for_sentence(
                    tokenizer, model, line.rstrip(), layer)
                vectors_np = vectors_torch.numpy()
                dim = vectors_np.shape[1]

                vectors_squeezed = []
                words_squeezed = []
                current_vector = None
                current_word = ""
                current_vector_members = 0
                for token, vec in zip(tokenized, vectors_np):
                    if token.startswith("##"):
                        if current_vector is None:
                            current_vector = np.zeros((dim,))
                        current_vector += vec
                        current_word += token[2:]
                        current_vector_members += 1
                    else:
                        if current_vector is not None:
                            vectors_squeezed.append(
                                current_vector / current_vector_members)
                            words_squeezed.append(current_word)
                        current_vector = vec
                        current_word = token
                        current_vector_members = 1

                vectors_squeezed.append(
                    current_vector / current_vector_members)
                words_squeezed.append(current_word)

                yield np.stack(vectors_squeezed), words_squeezed


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
        "bert_model",
        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
            "bert-base-multilingual-cased", "bert-base-multilingual-uncased", "bert-base-chinese"],
        help="Variant of pre-trained model.")
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
    print("Data loaded.")

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
                aligned_formatted = [f"{tgt_tok[j]} ({j})" for j in aligned_indices]
                print(f"{i:2d}: {token} -- {', '.join(aligned_formatted)}")
            print()
        else:
            print(edge_cover(distance))


if __name__ == "__main__":
    main()
