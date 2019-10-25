#!/usr/bin/env python
# coding: utf-8

"""Evaluate clustering of centroid w.r.t language families."""

import argparse
from collections import defaultdict


import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.optimize import linear_sum_assignment

from utils import get_lng_database


LNG_INFO = get_lng_database()

def compute_homogenity(classes1, classes2):
    homogenities = []
    for group in classes1:
        max_in_second = max(
            sum(x in group2 for x in group) for group2 in classes2)
        homogenities.append(max_in_second / len(group))
    return sum(homogenities) / len(homogenities)


def score(classes1, classes2):
    mutual_scores = []
    for group1 in classes1:
        row = []
        for group2 in classes2:
            in_both = len(group1.intersection(group2))
            if in_both == 0:
                row.append(0)
            else:
                row.append(2 * in_both / (len(group1) + len(group2)))
        mutual_scores.append(row)
    mutual_scores = np.array(mutual_scores)
    mapping = linear_sum_assignment(1 - mutual_scores)
    return np.mean(mutual_scores[mapping])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "centroids", type=argparse.FileType('rb'),
        help=".npz file with saved centroids.")
    args = parser.parse_args()

    np_file = np.load(args.centroids)
    languages = np_file["languages"]
    centroids = np_file["centroids"]

    families_dict = defaultdict(list)
    for lng, info in LNG_INFO.items():
        families_dict[info["genus"]].append(lng)
    families = [
        set(family) for family in families_dict.values() if len(family) > 2]

    interesting_languages = [lng for family in families for lng in family]
    lng2idx = {lng: i for i, lng in enumerate(interesting_languages)}
    centroids = centroids[[l in interesting_languages for l in languages]]
    family_idx = [set(lng2idx[lng] for lng in family) for family in families]

    tree = hac.linkage(centroids, method='single', metric='cosine')
    clustered = hac.fcluster(tree, len(families), criterion='maxclust')

    clusters = [
        set(np.where(clustered == i)[0].tolist())
        for i in range(1, len(families) + 1)]

    homogenity = compute_homogenity(clusters, family_idx)
    completeness = compute_homogenity(family_idx, clusters)
    v_measure = 2 * homogenity * completeness / (homogenity + completeness)

    print(f"Homogenity:   {100 * homogenity:.2f}")
    print(f"Completeness: {100 * completeness:.2f}")
    print(f"V-Measure:    {100 * v_measure:.2f}")


if __name__ == "__main__":
    main()
