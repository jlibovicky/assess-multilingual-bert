#!/usr/bin/env python
# coding: utf-8

"""Train language ID with BERT."""

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

from utils import text_data_generator, batch_generator, get_repr_from_layer, load_bert

logging.basicConfig(level=logging.INFO)

LANGUAGES = [
    "af", "an", "ar", "ast", "azb", "az", "bar", "ba", "be", "bg", "bn", "bpy",
    "br", "bs", "ca", "ceb", "ce", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fr", "fy", "ga", "gl", "gu", "he", "hi", "hr",
    "ht", "hu", "hy", "id", "io", "is", "it", "ja", "jv", "ka", "kk", "kn",
    "ko", "ky", "la", "lb", "lmo", "lt", "lv", "mg", "min", "mk", "ml", "mr",
    "ms", "my", "nb", "nds", "ne", "new", "nl", "nn", "no", "oc", "pa", "pl",
    "pms", "pnb", "pt", "ro", "ru", "scn", "sco", "sh", "sk", "sl", "sq", "sr",
    "su", "sv", "sw", "ta", "te", "tg", "tl", "tr", "tt", "uk", "ur", "uz",
    "vi", "vo", "war", "yo", "zh"]


LNG2IDX = {lng: i for i, lng in enumerate(LANGUAGES)}


def lng_data_generator(path, epochs=1):
    for _ in range(epochs):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                lng = line.strip()
                lng_id = LNG2IDX[lng]
                yield torch.tensor(lng_id)


def get_centroids(device, model, data, labels, layer, mean_pool=False):
    """Get language centeroids based on labels."""

    labels = torch.cat(labels).to(device)
    text_repr = torch.cat([
        get_repr_from_layer(model, d.to(device), layer, mean_pool=mean_pool)
        for d in data])
    centroids = torch.zeros((len(LANGUAGES), text_repr.size(1)))
    for i, _ in enumerate(LANGUAGES):
        centroids[i] = text_repr[labels == i].mean(0)

    return centroids


def load_and_batch_data(txt, lng, tokenizer, batch_size=32, epochs=1):
    text_batches = batch_generator(
        text_data_generator(
            txt, tokenizer, epochs=epochs, max_len=110),
        size=batch_size, padding=True)
    lng_batches = batch_generator(
        lng_data_generator(lng, epochs=epochs), size=batch_size, padding=False)
    return zip(text_batches, lng_batches)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bert_model", type=str, help="Variant of pre-trained model.")
    parser.add_argument(
        "layer", type=int,
        help="Layer from of layer from which the representation is taken.")
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
        "--hidden", default=None, type=int,
        help="Size of the hidden classification layer.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--test-output", type=str, default=None,
        help="Output for example classification.")
    parser.add_argument(
        "--skip-tokenization", default=False, action="store_true",
        help="Only split on spaces, skip wordpieces.")
    parser.add_argument(
        "--mean-pool", default=False, action="store_true",
        help="If true, use mean-pooling instead of [CLS] vecotr.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="Center languages to be around coordinate origin.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model, model_dim, _ = load_bert(
        args.bert_model, device)

    if args.layer < -1:
        print("Layer index cannot be negative.")
        exit(1)
    num_layers = len(model.encoder.layer)
    if args.layer >= num_layers:
        print(f"Model only has {num_layers} layers, {args.layer} is too much.")
        exit(1)

    train_batches = load_and_batch_data(
        args.train_data_txt, args.train_data_lng, tokenizer,
        batch_size=32, epochs=1000)
    print("Train data iterator initialized.")

    centroids = None
    if args.center_lng:
        print("Estimating language centroids.")
        with torch.no_grad():
            texts, labels = [], []
            for _, (txt, lab) in zip(range(100), train_batches):
                texts.append(txt)
                labels.append(lab)
            centroids = get_centroids(
                device, model, texts, labels,
                args.layer, mean_pool=args.mean_pool)
        centroids = centroids.to(device)

    print("Loading validation data.")
    val_batches_raw = list(load_and_batch_data(
        args.val_data_txt, args.val_data_lng, tokenizer,
        batch_size=32, epochs=1))[:2]
    print("Validation data loaded in memory, pre-computing BERT.")
    val_batches = []
    with torch.no_grad():
        for tokens, lng in val_batches_raw:
            bert_features = get_repr_from_layer(
                model, tokens.to(device), args.layer, args.mean_pool).cpu()
            val_batches.append((bert_features, lng))

    print("Loading test data.")
    test_batches_raw = list(load_and_batch_data(
        args.test_data_txt, args.test_data_lng, tokenizer,
        batch_size=32, epochs=1))[:2]
    print("Test data loaded in memory, pre-computing BERT.")
    test_batches = []
    with torch.no_grad():
        for tokens, lng in test_batches_raw:
            bert_features = get_repr_from_layer(
                model, tokens.to(device), args.layer, args.mean_pool).cpu()
            test_batches.append((bert_features, lng))
    print()

    test_accuracies = []
    all_test_outputs = []

    for exp_no in range(5):
        print(f"Starting experiment no {exp_no + 1}")
        print(f"------------------------------------")
        if args.hidden is None:
            classifier = nn.Linear(model_dim, len(LANGUAGES))
        else:
            classifier = nn.Sequential(
                nn.Linear(model_dim, args.hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(args.hidden, len(LANGUAGES)))
        classifier = classifier.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

        def evaluate(data_batches):
            classifier.eval()
            with torch.no_grad():
                running_val_loss = 0.
                running_val_acc = 0.
                val_count = 0
                outputs = []

                for bert_features, lng in data_batches:
                    bert_features, lng = (
                        bert_features.to(device), lng.to(device))
                    batch_size = bert_features.size(0)

                    if centroids is not None:
                        bert_features = bert_features - centroids[lng]
                    prediction = classifier(bert_features)

                    batch_loss = criterion(prediction, lng)

                    predicted_lng = prediction.max(-1)[1]
                    batch_accuracy = torch.sum((predicted_lng == lng).float())

                    running_val_loss += batch_size * batch_loss.cpu().numpy().tolist()
                    running_val_acc += batch_accuracy.cpu().numpy().tolist()
                    val_count += batch_size

                    outputs.extend(predicted_lng.cpu().numpy().tolist())

                val_loss = running_val_loss / val_count
                accuracy = running_val_acc / val_count

            return val_loss, accuracy, outputs

        best_accuracy = 0.0
        no_improvement = 0
        learning_rate_decreased = 0
        learning_rate = 1e-3

        for i, (sentences, lng) in enumerate(train_batches):
            try:
                classifier.train()
                optimizer.zero_grad()
                sentences, lng = sentences.to(device), lng.to(device)
                bert_features = get_repr_from_layer(
                    model, sentences, args.layer)

                if centroids is not None:
                    with torch.no_grad():
                        bert_features = bert_features - centroids[lng]

                prediction = classifier(bert_features)

                loss = criterion(prediction, lng)

                loss.backward()
                optimizer.step()

                if i % 10 == 9:
                    print(f"loss: {loss.cpu().detach().numpy().tolist():5g}")

                if i % 50 == 49:
                    print()
                    val_loss, accuracy, _ = evaluate(val_batches)

                    print("Validation: "
                          f"loss: {val_loss:5g}, "
                          f"accuracy: {accuracy:5g}")

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        no_improvement = 0
                    else:
                        no_improvement += 1

                    if no_improvement >= 5:
                        if learning_rate_decreased >= 5:
                            print(
                                "Learning rate decreased five times, ending.")
                            break

                        learning_rate /= 2
                        print(f"Decreasing learning rate to {learning_rate}.")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                        learning_rate_decreased += 1
                        no_improvement = 0

                    print()
            except KeyboardInterrupt:
                break

        model.eval()
        test_loss, test_accuracy, test_outputs = evaluate(test_batches)
        print()
        print("Testing:")
        print(f"test loss: {test_loss:5g}, "
              f"test accuracy: {test_accuracy:5g}")

        test_accuracies.append(test_accuracy)

        this_test_outputs = []
        for lng_prediction in test_outputs:
            this_test_outputs.append(LANGUAGES[lng_prediction])
        all_test_outputs.append(this_test_outputs)

    print()
    print("===============================================")
    print("All experiments done.")
    print("===============================================")
    print(f"Mean test accuracy {np.mean(test_accuracies)}")
    print(f"Mean test stdev    {np.std(test_accuracies)}")

    best_exp_id = np.argmax(test_accuracies)

    if args.test_output is not None:
        with open(args.test_output, 'w') as f_out:
            for prediction in test_outputs[best_exp_id]:
                print(prediction, file=f_out)


if __name__ == "__main__":
    main()
