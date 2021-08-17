#!/usr/bin/env python

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import load_word_embeddings, mean_word_embedding

def load_dataset(txt_file, lng_file, all_embeddings, lng2idx):
    representations = []
    targets = []

    with open(txt_file) as f_txt, open(lng_file) as f_lng:
        for sentence, lng in zip(f_txt, f_lng):
            lng = lng.strip()
            vector = mean_word_embedding(
                all_embeddings[lng], sentence.strip(), lng)

            if vector.shape == tuple():
                continue

            representations.append(vector)
            targets.append(lng2idx[lng])

    return np.stack(representations), np.array(targets)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "embeddings_prefix", type=str, help="Directory with word embeddings.")
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
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--save-model", type=str, help="Path where to save the best model.")
    parser.add_argument(
        "--save-centroids", type=str, help="Path to save language centroids.")
    parser.add_argument(
        "--test-output", type=str, default=None,
        help="Output for example classification.")
    parser.add_argument(
        "--center-lng", default=False, action="store_true",
        help="Center languages to be around coordinate origin.")
    args = parser.parse_args()

    with open(args.languages) as f_lang:
        languages = [line.strip() for line in f_lang]
    lng2idx = {lng: i for i, lng in enumerate(languages)}

    print("Loading embeddings.")
    all_embeddings = {
        lng: load_word_embeddings(f"{args.embeddings_prefix}/{lng}.vec")
        for lng in languages}

    print("Loading training data.")
    train_repr, train_tgt = load_dataset(
        args.train_data_txt, args.train_data_lng, all_embeddings, lng2idx)
    print("Loading test data.")
    test_repr, test_tgt = load_dataset(
        args.test_data_txt, args.test_data_lng, all_embeddings, lng2idx)

    if args.center_lng:
        centroids = np.stack([
            np.mean(train_repr[train_tgt == i], axis=0)
            for i in range(len(all_embeddings))])
        train_repr = train_repr - centroids[train_tgt]
        test_repr = test_repr - centroids[test_tgt]

    model = LogisticRegression()
    model.fit(train_repr, train_tgt)

    test_prediction = model.predict(test_repr)

    accuracy = np.mean(test_prediction == test_tgt)
    print(accuracy)


if __name__ == "__main__":
    main()
