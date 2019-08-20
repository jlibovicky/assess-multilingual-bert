#!/usr/bin/env python

"""Probe multilingual BERT on cross-lingual retrieval."""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

import logging
logging.basicConfig(level=logging.INFO)


def data_generator(path, tokenizer):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()

            # 512 is the maximum input size of BERT
            tokens = tokenizer.tokenize(sentence)
            tokenized = ["[CLS]"] + tokens[:510] + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokenized)
            yield torch.tensor(token_ids)


def pad_sentences(sentences):
    max_len = max(ex.size(0) for ex in sentences)
    padded_batch = torch.zeros(len(sentences), max_len, dtype=torch.int64)
    for i, ex in enumerate(sentences):
        padded_batch[i,:ex.size(0)] = ex
    return padded_batch


def batch_generator(generator, size):
    sentences = []

    for sentence in generator:
        sentences.append(sentence)

        if len(sentences) > size:
            yield pad_sentences(sentences)
            sentences = []
    if sentences:
        yield pad_sentences(sentences)


def get_repr_from_layer(model, data, layer, mean_pool=False):
    if layer >= 0:
        layer_output = model(data, torch.zeros_like(data))[0][layer]
        if mean_pool:
            mask = (data != 0).float().unsqueeze(2)
            lengths = mask.long().sum(1)

            # Mask out [CLS] and [SEP] symbols as well.
            mask[:,lengths - 1] = 0
            mask[:,0] = 0
            return (layer_output * mask).sum(1) / mask.sum(1)
        else:
            return layer_output[:, 0]
    elif layer == -1:
        if mean_pool:
            raise ValueError(f"Cannot mean-pool the default vector.")
        return model(data, torch.zeros_like(data))[1]
    else:
        raise ValueError(f"Invalid layer {layer}.")


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
        "bert_model",
        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
            "bert-base-multilingual-cased", "bert-base-multilingual-uncased", "bert-base-chinese"],
        help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
    parser.add_argument(
        "data", type=str, help="Sentences with language for training.", nargs="+")
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
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    distance_fn = None
    if args.distance == "cosine":
        distance_fn = cosine_distances
    elif args.distance == "euklid":
        distance_fn = euklid_distances
    else:
        raise ValueError("Unknown distance function.")


    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    model = BertModel.from_pretrained(args.bert_model).to(device)
    model.eval()

    representations = []

    with torch.no_grad():
        for text_file in args.data:
            print(f"Processing {text_file}")
            vectors = [
                get_repr_from_layer(
                    model, sentence_tensor, args.layer,
                    mean_pool=args.mean_pool)
                for sentence_tensor in batch_generator(
                    data_generator(text_file, tokenizer), 64)]

            lng_repr = torch.cat(vectors, dim=0)
            if args.center_lng:
                lng_repr = lng_repr - lng_repr.mean(0, keepdim=True)

            representations.append(lng_repr)

        data_len = representations[0].shape[0]
        assert(all(r.shape[0] == data_len for r in representations))
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
