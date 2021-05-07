#!/usr/bin/env python

"""Learn a projection from one language space to another."""

import argparse
import logging
import sys

import joblib
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

from utils import (
    text_data_generator, batch_generator, get_repr_from_layer, load_bert)

logging.basicConfig(level=logging.INFO)


def repr_for_text_file(text_file, model, tokenizer, layer, mean_pool):
    with torch.no_grad():
        vectors = [
            get_repr_from_layer(
                model, sentence_tensor, layer, tokenizer.pad_token_id,
                mean_pool=mean_pool)
            for sentence_tensor in batch_generator(
                text_data_generator(text_file, tokenizer), 64, tokenizer)]
        return torch.cat(vectors, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
    parser.add_argument(
        "data_lng1", type=str,
        help="Sentences with language for training.")
    parser.add_argument(
        "data_lng2", type=str,
        help="Sentences with language for training.")
    parser.add_argument(
        "save_model", type=str, help="Path to the saved model.")
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vector.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model = load_bert(args.bert_model, device)[:2]

    print(f"Loading representation for {args.data_lng1}", file=sys.stderr)
    lng1_repr = repr_for_text_file(
        args.data_lng1, model, tokenizer, args.layer, args.mean_pool)
    print(f"Loading representation for {args.data_lng2}", file=sys.stderr)
    lng2_repr = repr_for_text_file(
        args.data_lng2, model, tokenizer, args.layer, args.mean_pool)
    print("BERT representations loaded.", file=sys.stderr)

    print("Fitting the projection.", file=sys.stderr)
    model = LinearRegression()
    model.fit(lng1_repr, lng2_repr)
    print("Done, saving model.", file=sys.stderr)

    joblib.dump(model, args.save_model)

if __name__ == "__main__":
    main()
