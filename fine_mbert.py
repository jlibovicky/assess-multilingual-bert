#!/usr/bin/env python
# coding: utf-8

"""Train language ID with BERT."""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_transformers import BertForMaskedLM, BertTokenizer
from pytorch_revgrad import RevGrad

from lang_id import load_and_batch_data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "languages", type=str,
        help="File with a list of languages.")
    parser.add_argument(
        "train_data_txt", type=str, help="Training sentences.")
    parser.add_argument(
        "train_data_lng", type=str,
        help="Language codes for training sentences.")
    parser.add_argument(
        "val_data_txt", type=str, help="Validation sentences.")
    parser.add_argument(
        "val_data_lng", type=str,
        help="Language codes for validation sentences.")
    parser.add_argument(
        "test_data_txt", type=str, help="Test sentences.")
    parser.add_argument(
        "test_data_lng", type=str, help="Language codes for test sentences.")
    parser.add_argument(
        "--hidden", default=1024, type=int,
        help="Size of the hidden classification layer.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--save-model", type=str, help="Path where to save the best model.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--update-every-batch", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--bert-lr", type=float, default=1e-7,
        help="Learning rate for finetuning BERT.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.languages) as f_lang:
        languages = [line.strip() for line in f_lang]
    lng2idx = {lng: i for i, lng in enumerate(languages)}

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.bert_model.endswith("-uncased"))
    model = BertForMaskedLM.from_pretrained(
        args.bert_model, output_hidden_states=True).to(device)

    model_dim = model.bert.encoder.layer[-1].output.dense.out_features

    train_batches = load_and_batch_data(
        args.train_data_txt, args.train_data_lng, tokenizer,
        lng2idx, batch_size=args.batch_size, epochs=args.epochs)
    print("Train data iterator initialized.")

    def get_classifier():
        return nn.Sequential(
            RevGrad(),
            nn.Linear(model_dim, args.hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden, len(languages))).to(device)

    cls_classifiers = [get_classifier() for _ in range(12)]
    state_classifiers = [get_classifier() for _ in range(12)]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": 1e-6}] + [
            {"params": cls.parameters()} for cls
            in cls_classifiers + state_classifiers],
        lr=1e-4)

    for i, (sentences, lng) in enumerate(train_batches):
        try:
            model.train()
            for cls in cls_classifiers + state_classifiers:
                cls.train()

            sentences, lng = sentences.to(device), lng.to(device)

            # Len mask not to include padded tokens
            len_mask = (sentences != 0).float()

            # This is the mask that says where we do the BERT magic, i.e., (1)
            # these are the tokens where we want to predict what word actually
            # should on the input and (2) these are the tokens we want to mess
            # up the input .. but be careful, we don't want to predict the
            # padding.
            randomized_mask = (
                torch.zeros_like(len_mask).uniform_() * len_mask) > 0.85

            # This the prediction targe. Predict the real word on the
            # randomized places, -1 elswhere
            lm_target = torch.where(
                randomized_mask, sentences, -torch.ones_like(len_mask).long())

            # Create an "alternative input":
            #  * 80% [MASK];
            #  * 10% random words;
            #  * 10% original words.
            alternative_input = (
                torch.ones_like(sentences) * tokenizer.mask_token_id)
            random_values = torch.zeros_like(len_mask).uniform_()

            alternative_input = torch.where(
                random_values < 0.1, # with 10% probability
                torch.zeros_like(sentences).random_(0, len(tokenizer.vocab)),
                alternative_input) # otherwise [MASK]

            alternative_input = torch.where(
                random_values > 0.9, # with 10% probability
                sentences, # put the original token
                alternative_input) # otherwise keep the alternative

            bert_input = torch.where(
                randomized_mask, alternative_input, sentences)

            bert_loss, _, hidden_states = model(
                bert_input, attention_mask=len_mask,
                masked_lm_labels=lm_target)

            cls_repr = [states[:, 0] for states in hidden_states[1:]]
            state_repr = [
                (states[:, 1:] * len_mask[:, 1:].unsqueeze(2)).sum(1)
                / (len_mask.sum(1) - 1).unsqueeze(1)
                for states in hidden_states[1:]]

            cls_losses = [
                criterion(cls(rep), lng) for cls, rep
                in zip(cls_classifiers, cls_repr)]
            mean_losses = [
                criterion(cls(rep), lng) for cls, rep
                in zip(state_classifiers, state_repr)]

            loss = sum(cls_losses) + sum(mean_losses) + 100 * bert_loss

            loss.backward()
            if i % args.update_every_batch == args.update_every_batch - 1:
                optimizer.step()
                optimizer.zero_grad()

                lngid_loss = (sum(cls_losses) + sum(mean_losses)) / 24

                def loss_to_pyfloat(loss):
                    return loss.cpu().detach().numpy().tolist()

                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S")}, {i + 1} steps:   '
                    f"BERT loss: {loss_to_pyfloat(bert_loss):5g}  "
                    f"lngid loss: {loss_to_pyfloat(lngid_loss):5g}")

        except KeyboardInterrupt:
            print("Training interrupted by user")
            break

    if args.save_model is not None:
        os.makedirs(args.save_model)
        model.eval()
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == "__main__":
    main()
