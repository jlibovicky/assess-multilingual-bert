#!/usr/bin/env python
# coding: utf-8

"""Make TSNE plot of language centroids."""

import argparse

import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "centroids", type=argparse.FileType('rb'),
        help=".npz file with saved centroids.")
    parser.add_argument(
        "output_img", type=argparse.FileType('wb'),
        help="Plot with tSNE output.")
    args = parser.parse_args()

    np_file = np.load(args.centroids)
    languages = np_file["languages"]
    centroids = np_file["centroids"]

    tsne = TSNE(n_components=2, random_state=0)
    centroids_2d = tsne.fit_transform(centroids)

    plt.figure(figsize=(16, 16))
    plt.title(f"tSNE of '{args.centroids.name}'")
    for name, point in zip(languages, centroids_2d):
        plt.scatter(point[0], point[1], s=0)
        plt.annotate(
            name, xy=point,
            xytext=(5, 2),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox={"boxstyle": "square,pad=0.3",
                  "fc": "lightgray"})
    plt.show()
    plt.savefig(args.output_img)


if __name__ == "__main__":
    main()
