#!/usr/bin/env python
# coding: utf-8

"""Train language ID with BERT."""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel

import logging
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


def data_generator(path, tokenizer, skip_tokenization, epochs=1):
    for _ in range(epochs) :
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' not in line:
                    print(f"Warning: line without \\t: '{line.strip()}'")
                    continue
                try:
                    sentence, lng = line.strip().split("\t")
                except ValueError:
                    continue
                lng_id = LNG2IDX[lng]

                # 512 is the maximum input size of BERT
                if skip_tokenization:
                    tokens = sentence.strip().split(" ")
                else:
                    tokens = tokenizer.tokenize(sentence.strip())
                tokenized = ["[CLS]"] + tokens[:100] + ["[SEP]"]
                token_ids = tokenizer.convert_tokens_to_ids(tokenized)
                yield torch.tensor(token_ids), torch.tensor(lng_id)


def pad_sentences(sentences):
    max_len = max(ex.size(0) for ex in sentences)
    padded_batch = torch.zeros(len(sentences), max_len, dtype=torch.int64)
    for i, ex in enumerate(sentences):
        padded_batch[i,:ex.size(0)] = ex
    return padded_batch


def batch_generator(generator, size):
    sentences = []
    languages = []

    for sentence, lng in generator:
        sentences.append(sentence)
        languages.append(lng)

        if len(sentences) > size:
            yield pad_sentences(sentences), torch.stack(languages)
            sentences = []
            languages = []
    if sentences:
        yield pad_sentences(sentences), torch.stack(languages)


def get_repr_from_layer(model, data, layer, mean_pool=False):
    if layer >= 0:
        layer_output = model(data, torch.zeros_like(data))[0][layer]
        if mean_pool:
            mask = (data != 0).float().unsqueeze(2)
            return (layer_output * mask).sum(1) / mask.sum(1)
        else:
            return layer_output[:, 0]
    elif layer == -1:
        if mean_pool:
            raise ValueError(f"Cannot mean-pool the default vector.")
        return model(data, torch.zeros_like(data))[1]
    else:
        raise ValueError(f"Invalid layer {layer}.")


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

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.bert_model.endswith("-uncased"))
    model = BertModel.from_pretrained(args.bert_model).to(device)
    model.eval()

    if args.layer < -1:
        print("Layer index cannot be negative.")
        exit(1)
    num_layers = len(model.encoder.layer)
    if args.layer >= num_layers:
        print(f"Model only has {num_layers} layers, {args.layer} is too much.")
        exit(1)

    model_dim = model.encoder.layer[args.layer].output.dense.out_features

    vocab_size = model.embeddings.word_embeddings.weight.size(0)

    train_data = data_generator(args.train_data, tokenizer, args.skip_tokenization, epochs=1000)
    train_batches = batch_generator(train_data, 32)
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

    print("Loading valiation data.")
    val_data = data_generator(args.val_data, tokenizer, args.skip_tokenization)
    val_batches_raw = list(batch_generator(val_data, 32))
    print("Validation data loaded in memory, pre-computing BERT.")
    val_batches = []
    with torch.no_grad():
        for tokens, lng in val_batches_raw:
            bert_features = get_repr_from_layer(
                model, tokens.to(device), args.layer, args.mean_pool).cpu()
            val_batches.append((bert_features, lng))

    print("Loading test data.")
    test_data = data_generator(args.test_data, tokenizer, args.skip_tokenization)
    test_batches_raw = list(batch_generator(test_data, 32))
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
                nn.Linear(model_dim,  args.hidden),
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
                    bert_features, lng = bert_features.to(device), lng.to(device)
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
        lr_decreased = 0
        lr = 1e-3

        for i, (sentences, lng) in enumerate(train_batches):
            try:
                classifier.train()
                optimizer.zero_grad()
                sentences, lng = sentences.to(device), lng.to(device)
                bert_features = get_repr_from_layer(model, sentences, args.layer)

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
