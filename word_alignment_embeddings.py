#!/usr/bin/env python
# coding: utf-8

"""Do word alignment by minimum weighted edge cover."""

import argparse
import logging
import sys

import torch

from word_alignment import align, center
from utils import load_word_embeddings, word_embeddings_for_file

logging.basicConfig(level=logging.INFO)


def load_data(
        src, tgt, src_emb, tgt_emb, src_lng, tgt_lng, center_lng):
    print(f"Loading src: {src} ... ", file=sys.stderr, end="", flush=True)
    src_embeddings = load_word_embeddings(src_emb)
    print(" embeddings ... ", end="", file=sys.stderr)
    src_repr = word_embeddings_for_file(
        src, src_embeddings, src_lng, mean_pool=False,
        skip_tokenization=True)
    print("Done", file=sys.stderr)

    print(f"Loading tgt: {tgt} ... ", file=sys.stderr, end="", flush=True)
    tgt_embeddings = load_word_embeddings(tgt_emb)
    print(" embeddings ... ", end="", file=sys.stderr)
    tgt_repr = word_embeddings_for_file(
        tgt, tgt_embeddings, tgt_lng, mean_pool=False,
        skip_tokenization=True)
    print("Done", file=sys.stderr)

    if center_lng:
        print("Centering data.", file=sys.stderr)
        src_repr, tgt_repr = center(src_repr), center(tgt_repr)

    return src_repr, tgt_repr


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "src", type=str, help="Sentences in source language.")
    parser.add_argument(
        "tgt", type=str, help="Sentences in the target language.")
    parser.add_argument(
        "src_emb", type=str, help="Source language word embeddings.")
    parser.add_argument(
        "tgt_emb", type=str, help="Target language word embeddings.")
    parser.add_argument(
        "src_lng", type=str, help="Source language code.")
    parser.add_argument(
        "tgt_lng", type=str, help="Target language code.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="If true, center representations first.")
    parser.add_argument(
        "--reordering-penalty", default=1e-5, type=float,
        help="Penalty for long-distance alignment added to cost.")
    parser.add_argument(
        "--verbose", default=False, action="store_true",
        help="If true, print the actual alignment.")
    parser.add_argument(
        "--save-projection", type=str, default=None,
        help="Location to save the word projection.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    proj = None

    print("Loading test data.", file=sys.stderr)
    src_repr, tgt_repr = load_data(
        args.src, args.tgt, args.src_emb, args.tgt_emb,
        args.src_lng, args.tgt_lng, args.center_lng)

    for (src_mat, src_tok), (tgt_mat, tgt_tok) in zip(src_repr, tgt_repr):
        alignment = align(src_mat, tgt_mat, args.reordering_penalty, proj)

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
