#!/usr/bin/env python
# coding: utf-8

"""Compute attention head entropies per language."""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

import logging
logging.basicConfig(level=logging.INFO)


def text_data_generator(path, tokenizer):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()

            # 512 is the maximum input size of BERT
            tokens = tokenizer.tokenize(sentence)
            tokenized = ["[CLS]"] + tokens[:510] + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokenized)
            yield torch.tensor(token_ids)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model",
        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
            "bert-base-multilingual-cased", "bert-base-multilingual-uncased", "bert-base-chinese"],
        help="Variant of pre-trained model.")
    parser.add_argument(
        "language_data", nargs="+", type=str,
        help="Files with data, name of the file is language code.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    model = BertModel.from_pretrained(
        args.bert_model,
        output_attentions=True,
        keep_multihead_output=True).to(device)
    model.eval()

    languages = []
    entropies = []

    with torch.no_grad():
        for input_file in args.language_data:
            lng_code = input_file.split("/")[-1][:-4]
            print(f"Working on {lng_code}")

            entropies_sums = None
            sentence_count = 0

            for sentence_tensor in text_data_generator(input_file, tokenizer):
                sentence_count += 1
                layer_attentions = model(sentence_tensor.unsqueeze(0))[0]
                head_count = layer_attentions[0].shape[1]

                if entropies_sums is None:
                    entropies_sums = np.zeros(
                        len(layer_attentions) * head_count)

                head_id = 0
                for att_matrices in layer_attentions:
                    for matrix in att_matrices.squeeze(0):
                        entropy = -torch.mean((matrix * torch.log(matrix + 1e-9)).sum(1))
                        entropies_sums[head_id] += entropy.cpu().numpy()
                        head_id += 1

                if sentence_count >= args.limit:
                    break

            languages.append(lng_code)
            entropies.append(entropies_sums / sentence_count)

    for lng, entropy in zip(languages, entropies):
        formatted_ent = "\t".join([f"{e:.5f}" for e in entropy])
        print(f"{lng}\t{formatted_ent}")


if __name__ == "__main__":
    main()
