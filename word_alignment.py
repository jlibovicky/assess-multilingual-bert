#!/usr/bin/env python
# coding: utf-8

"""Do word alignment by minimum weighted edge cover."""

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
from sklearn.neural_network import MLPRegressor

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
                    if len(bert_split) >= 1:
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


def apply_sklearn_proj(representations, model_path):
    print("Projecting representations.", file=sys.stderr)
    model = joblib.load(model_path)
    for vectors, tokens in representations:
        yield model.predict(vectors), tokens


def reordering_penalty(src_size, tgt_size):
    """Penalty for reordering 0-1 based on relative token distance."""
    penalty_matrix = np.zeros((src_size, tgt_size))
    max_size = max(src_size, tgt_size)
    for i in range(src_size):
        for j in range(tgt_size):
            penalty_matrix[i, j] = abs(i - j) / max_size
    return penalty_matrix


def align(src_mat, tgt_mat, penalty, proj=None):
    if proj is not None:
        src_mat = proj.predict(src_mat)

    # Cosine in (-1, 1) and MWEC requires positive weights => 2 -
    distance = 2 - (
        np.dot(src_mat, tgt_mat.T)
        / np.expand_dims(np.linalg.norm(src_mat, axis=1), 1)
        / np.expand_dims(np.linalg.norm(tgt_mat, axis=1), 0))

    if penalty > 0:
        distance += (
            penalty * reordering_penalty(*distance.shape))
    return edge_cover(distance)


def center(representations):
    vec_center = np.mean(
        np.concatenate(
            [mat for mat, txt in representations]), 0, keepdims=True)
    return [(mat - vec_center, txt) for mat, txt in representations]


def em_step(src_repr, tgt_repr, penalty, orig_proj):
    src_vectors = []
    tgt_vectors = []

    print("Computing alignment ... ", end="", file=sys.stderr, flush=True)
    for (src_mat, _), (tgt_mat, _) in zip(src_repr, tgt_repr):
        for i, j in align(src_mat, tgt_mat, penalty, orig_proj):
            src_vectors.append(src_mat[i])
            tgt_vectors.append(tgt_mat[j])
    print("Done", file=sys.stderr)

    new_proj = MLPRegressor((768), early_stopping=True) #LinearRegression()
    print("Fitting regression ... ", end="", file=sys.stderr, flush=True)
    new_proj.fit(src_vectors, tgt_vectors)
    print("Done", file=sys.stderr)

    return new_proj


def load_data(
        src, tgt, model, tokenizer, layer, center_lng, src_proj, tgt_proj):
    print(f"Loading src: {src} ... ", file=sys.stderr, end="", flush=True)
    src_repr = list(call_bert_and_collapse_tokens(
        src, model, tokenizer, layer))
    print("Done", file=sys.stderr)

    print(f"Loading tgt: {tgt} ... ", file=sys.stderr, end="", flush=True)
    tgt_repr = list(call_bert_and_collapse_tokens(
        tgt, model, tokenizer, layer))
    print("Done", file=sys.stderr)

    if center_lng:
        print("Centering data.", file=sys.stderr)
        src_repr, tgt_repr = center(src_repr), center(tgt_repr)
    if src_proj is not None:
        src_repr = list(apply_sklearn_proj(src_repr, src_proj))
    if tgt_proj is not None:
        tgt_repr = list(apply_sklearn_proj(tgt_repr, tgt_proj))

    return src_repr, tgt_repr


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
        "--src-proj", default=None, type=str,
        help="Sklearn projection of the source language.")
    parser.add_argument(
        "--tgt-proj", default=None, type=str,
        help="Sklearn projection of the target language.")
    parser.add_argument(
        "--reordering-penalty", default=1e-5, type=float,
        help="Penalty for long-distance alignment added to cost.")
    parser.add_argument(
        "--verbose", default=False, action="store_true",
        help="If true, print the actual alignment.")
    parser.add_argument(
        "--iterations", type=int, default=0,
        help="Number of EM iterations.")
    parser.add_argument(
        "--train-data", type=str, nargs=2, default=None,
        help="Training data for EM training.")
    parser.add_argument(
        "--save-projection", type=str, default=None,
        help="Location to save the word projection.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.center_lng and (
            args.src_proj is not None and args.tgt_proj is not None):
        print("You can either project or center "
              "the representations, not both.", file=sys.stderr)
        exit(1)

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    proj = None
    if args.iterations > 0:
        if args.train_data is None:
            print("You need to specify train data for EM training.",
                  file=sys.stderr)
            exit(1)

        print("Loading training data.", file=sys.stderr)
        train_src_repr, train_tgt_repr = load_data(
            args.train_data[0], args.train_data[1],
            model, tokenizer, args.layer,
            args.center_lng, args.src_proj, args.tgt_proj)

        for iteration in range(args.iterations):
            print(f"Iteration {iteration + 1}", file=sys.stderr)
            proj = em_step(
                train_src_repr, train_tgt_repr, args.reordering_penalty, proj)
            print("Done.", file=sys.stderr)

        if args.save_projection:
            joblib.dump(proj, args.save_projection)

    print("Loading test data.", file=sys.stderr)
    src_repr, tgt_repr = load_data(
        args.src, args.tgt, model, tokenizer, args.layer,
        args.center_lng, args.src_proj, args.tgt_proj)

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
