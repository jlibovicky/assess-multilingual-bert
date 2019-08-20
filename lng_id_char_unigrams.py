#!/usr/bin/env python
# coding: utf-8

"""Train language classifier based on char frequency."""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


LANGUAGES = [
    "af", "ar", "cs", "de", "es", "et", "eu", "fr", "hi", "hu", "is",
    "it", "ka", "ko", "nl", "no", "pt", "ru", "sk", "sl", "tl", "vi", "yo",
    "zh"]


LNG2IDX = {lng: i for i, lng in enumerate(LANGUAGES)}


def load_char_vocab(path, freq_limit=5):
    chars = []
    char2idx = {}
    with open(path) as f_vocab:
        for line in f_vocab:
            word, freq_str = line.strip().split("\t")
            freq = int(freq_str)
            if freq >= freq_limit:
                char2idx[word] = len(chars)
                chars.append(word)

    return chars, char2idx


def sentence2vec(sent, char2idx):
    vector = np.zeros(len(char2idx))
    for char in sent:
        if char in char2idx:
            vector[char2idx[char]] += 1
    return vector


def load_data_as_tensors(path, char2idx):
    sentences = []
    lngs = []

    with open(path) as f_data:
        for line in f_data:
            try:
                sentence, lng = line.strip().split("\t")
            except ValueError:
                continue

            sentences.append(sentence2vec(sentence, char2idx))
            lngs.append(LNG2IDX[lng])

    return torch.tensor(np.array(sentences), dtype=torch.float32), torch.tensor(np.array(lngs), dtype=torch.int64)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "char_vocabulary", type=str,
        help="Path to character vocabulary.")
    parser.add_argument(
        "train_data", type=str, help="Sentences with language for training.")
    parser.add_argument(
        "val_data", type=str, help="Sentences with language for validation.")
    parser.add_argument(
        "test_data", type=str, help="Sentences with language for testing.")
    parser.add_argument(
        "--hidden", default=None, type=int,
        help="Size of the hidden classification layer.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--test-output", type=str, default=None,
        help="Output for example classification.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chars, char2idx = load_char_vocab(args.char_vocabulary)
    print("Vocabulary loaded.")

    train_data, train_lng = load_data_as_tensors(args.train_data, char2idx)
    print("Train data loaded.")
    val_data, val_lng = load_data_as_tensors(args.val_data, char2idx)
    val_data, val_lng = val_data.to(device), val_lng.to(device)
    print("Val data loaded.")
    test_data, test_lng = load_data_as_tensors(args.test_data, char2idx)
    test_data, test_lng = test_data.to(device), test_lng.to(device)
    print("Test data loaded.")

    if args.hidden is None:
        classifier = nn.Linear(len(chars), len(LANGUAGES))
    else:
        classifier = nn.Sequential(
            nn.Linear(len(chars), args.hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden, len(LANGUAGES)))
    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    def evaluate(data, tgt):
        classifier.eval()
        with torch.no_grad():
            prediction = classifier(data)

            loss = criterion(prediction, tgt)

            predicted_lng = prediction.max(-1)[1]
            accuracy = torch.mean((predicted_lng == tgt).float())

        return loss, accuracy, predicted_lng

    best_accuracy = 0.0
    no_improvement = 0
    lr_decreased = 0
    lr = 1e-3

    for i, (sentences, lng) in enumerate(zip(train_data.split(32), train_lng.split(32))):
        try:
            classifier.train()
            optimizer.zero_grad()
            sentences, lng = sentences.to(device), lng.to(device)
            prediction = classifier(sentences)

            loss = criterion(prediction, lng)

            loss.backward()
            optimizer.step()

            if i % 10 == 9:
                print(f"loss: {loss.cpu().detach().numpy().tolist():5g}")

            if i % 100 == 99:
                print()
                val_loss, accuracy, _ = evaluate(val_data, val_lng)

                print("Validation: "
                      f"loss: {val_loss:5g}, "
                      f"accuracy: {accuracy:5g}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= 5:
                    if lr_decreased >= 5:
                        print("Learning rate decreased five times, ending.")
                        break

                    lr /= 2
                    print(f"Decreasing learning rate to {lr}.")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    lr_decreased += 1
                    no_improvement = 0

                print()
        except KeyboardInterrupt:
            break

    classifier.eval()
    test_loss, test_accuracy, test_outputs = evaluate(test_data, test_lng)
    print()
    print("Testing:")
    print(f"test loss: {test_loss:5g}, "
          f"test accuracy: {test_accuracy:5g}")

    if args.test_output is not None:
        with open(args.test_output, 'w') as f_out:
            for lng_prediction in test_outputs:
                print(LANGUAGES[lng_prediction], file=f_out)


if __name__ == "__main__":
    main()



