#!/usr/bin/env python
# coding: utf-8

"""Make TSNE plot of language centroids."""

import argparse

import numpy as np
import scipy.cluster.hierarchy as hac
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils import get_lng_database

LANGUAGE_FAMILIES = [
    "Romance", "Germanic", "Slavic", "Turkic", "Indic", "Malayo-Sumbawan",
    "Celtic", "Southern Dravidian", "Semitic"]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


FAMILY_COLORS = {
    family: color for family, color in zip(LANGUAGE_FAMILIES, COLORS)}

LNG_INFO = get_lng_database()


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

    plt.figure(figsize=(32, 16))
    plt.subplot(121)

    plt.title(f"tSNE of '{args.centroids.name}'")
    for name, point in zip(languages, centroids_2d):
        plt.scatter(point[0], point[1], s=0)

        color = "#ffffff"
        if "genus" in LNG_INFO[name]:
            family = LNG_INFO[name]["genus"]
            if family in FAMILY_COLORS:
                color = FAMILY_COLORS[family]

        plt.annotate(
            name, xy=point,
            xytext=(5, 2),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox={"boxstyle": "square,pad=0.3",
                  "fc": f"{color}aa"})

    for family, color in FAMILY_COLORS.items():
        family_languages = set(
            lng["name"] for lng in LNG_INFO.values()
            if "genus" in lng and lng["genus"] == family)

        lng_points = [
            point for name, point in zip(languages, centroids_2d)
            if name in family_languages]
        avg_point = (
            np.median(lng_points, axis=0) +
            np.mean(lng_points, axis=0)) / 2

        plt.annotate(
            family, xy=avg_point,
            xytext=(5, 2),
            textcoords='offset points',
            ha='center', va='center',
            color="white", weight='bold', size=18,
            backgroundcolor=color)

    plt.subplot(122)

    clustering = hac.linkage(centroids, method='complete', metric='cosine')
    link_colors = ["#aaaaaa"] * (2 * len(centroids) - 1)
    family_nodes = {}

    def get_family_for_index(index):
        if index < len(languages):
            return LNG_INFO[languages[index]]["genus"]
        return family_nodes.get(index)

    for i, (item1, item2, _, _) in enumerate(clustering):
        new_id = len(languages) + i
        item1, item2 = int(item1), int(item2)

        family1 = get_family_for_index(int(item1))
        family2 = get_family_for_index(int(item2))

        if family1 is None or family2 is None:
            continue

        if family1 == family2:
            family_nodes[new_id] = family1
            if family1 in FAMILY_COLORS:
                link_colors[new_id] = FAMILY_COLORS[family1]
            else:
                link_colors[new_id] = "black"

    hac.dendrogram(
        clustering, labels=languages, orientation='left',
        leaf_font_size=10, show_leaf_counts=True,
        link_color_func=lambda k: link_colors[k])

    ax = plt.gca()
    xlbls = ax.get_ymajorticklabels()
    for lbl in xlbls:
        lng_name = lbl.get_text()
        if "genus" in LNG_INFO[lng_name]:
            family = LNG_INFO[lng_name]["genus"]
            if family in FAMILY_COLORS:
                color = FAMILY_COLORS[family]
                lbl.set_bbox({
                    "boxstyle": "square,pad=0.",
                    "fc": f"{color}aa",
                    "ec": "none"})
                #lbl.set_backgroundcolor(color)
                #lbl.set_color(label_colors[lbl.get_text()])



    plt.show()
    plt.savefig(args.output_img)


if __name__ == "__main__":
    main()
