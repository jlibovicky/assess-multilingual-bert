#!/usr/bin/env python

"""Generate confusion matrix for language classification."""


import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


LANGUAGES = [
    "af", "ar", "cs", "de", "es", "et", "eu", "fr", "hi", "hu", "is",
    "it", "ka", "ko", "nl", "no", "pt", "ru", "sk", "sl", "tl", "vi", "yo",
    "zh"]

LNG2IDX = {lng: i for i, lng in enumerate(LANGUAGES)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "test_data", type=argparse.FileType("r"),
        help="Tab-separated test data.")
    parser.add_argument(
        "predictions", type=argparse.FileType("r"),
        help="Predictions made by the model.")
    args = parser.parse_args()

    true_labels = [
        LNG2IDX[line.strip().split("\t")[1]] for line in args.test_data]
    predicted_labels = [
        LNG2IDX[line.strip()] for line in args.predictions]

    matrix = confusion_matrix(true_labels, predicted_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    #im = ax.matshow(matrix)
    im1 = ax1.imshow(np.ma.array(matrix, mask=np.eye(matrix.shape[0])))
    fig.colorbar(im1, ax=ax1, orientation="horizontal", fraction=0.046, pad=0.04)
    ax1.set_ylim(-0.5, matrix.shape[1] - 0.5)

    ax1.set_xticks(np.arange(len(LANGUAGES)), minor=False)
    ax1.set_yticks(np.arange(len(LANGUAGES)), minor=False)
    ax1.set_xticklabels(LANGUAGES)
    ax1.set_yticklabels(LANGUAGES)

    ax1.set_xticks(np.arange(len(LANGUAGES)) + 0.5, minor=True)
    ax1.set_yticks(np.arange(len(LANGUAGES)) + 0.5, minor=True)
    ax1.yaxis.grid(True, which='minor', linestyle='-', linewidth=1)
    ax1.xaxis.grid(True, which='minor', linestyle='-', linewidth=1)

    im2 = ax2.imshow(np.ma.array(matrix, mask=1 - np.eye(matrix.shape[0])))
    fig.colorbar(im2, ax=ax2, orientation="horizontal", fraction=0.046, pad=0.04)
    ax2.set_ylim(-0.5, matrix.shape[1] - 0.5)

    ax2.set_xticks(np.arange(len(LANGUAGES)), minor=False)
    ax2.set_yticks(np.arange(len(LANGUAGES)), minor=False)
    ax2.set_xticklabels(LANGUAGES)
    ax2.set_yticklabels(LANGUAGES)

    ax2.set_xticks(np.arange(len(LANGUAGES)) + 0.5, minor=True)
    ax2.set_yticks(np.arange(len(LANGUAGES)) + 0.5, minor=True)
    ax2.yaxis.grid(True, which='minor', linestyle='-', linewidth=1)
    ax2.xaxis.grid(True, which='minor', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.savefig(f"confusion_matrix.pdf")


if __name__ == "__main__":
    main()
