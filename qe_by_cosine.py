#!/usr/bin/env python
# coding: utf-8

"""Compute centroids of BERT representations for all languages."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import data_generator, batch_generator, get_repr_from_layer

import logging
import sys
logging.basicConfig(level=logging.INFO)


def repr_for_txt_file(filename, tokenizer, model, device, layer, center_lng=True, mean_pool=True):
    print(f"Processing {filename}", file=sys.stderr)
    with torch.no_grad():
        vectors = [
            get_repr_from_layer(
                model, sentence_tensor.to(device), layer,
                mean_pool=mean_pool).cpu()
            for sentence_tensor in batch_generator(
                data_generator(filename, tokenizer), 32)]

        lng_repr = torch.cat(vectors, dim=0)
        if center_lng:
            lng_repr = lng_repr - lng_repr.mean(0, keepdim=True)
    return lng_repr


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
        "mt", type=str, help="Machine-translated sentences.")
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vecotr.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="If true, center representations first.")
    parser.add_argument(
        "--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.bert_model.endswith("-uncased"))
    model = BertModel.from_pretrained(args.bert_model).to(device)
    model.eval()

    src_repr = repr_for_txt_file(
        args.src, tokenizer, model, device, args.layer,
        center_lng=args.center_lng, mean_pool=args.mean_pool)
    mt_repr = repr_for_txt_file(
        args.mt, tokenizer, model, device, args.layer,
        center_lng=args.center_lng, mean_pool=args.mean_pool)

    src_norm = (src_repr * src_repr).sum(1).sqrt()
    mt_norm = (mt_repr * mt_repr).sum(1).sqrt()

    cosine = (src_repr * mt_repr).sum(1) / src_norm / mt_norm

    for num in cosine.cpu().detach().numpy():
        print(num)


if __name__ == "__main__":
    main()
