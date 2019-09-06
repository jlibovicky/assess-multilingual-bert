#!/usr/bin/python

"""
The algorithm is taken from:
Alexander Schrijver. Combinatorial Optimization: Polyhedra and Efficiency.
Volume A. Springer, 2003., p. 317

For a bipartite graph G its copy is made having the opposite direction of
edges. Except already existing edges, edes leading from each vertex to its
copy are added. Such edges are score given to two times the weight of
the minimum weighted edge from the vertex. Then the cheapest maximum matching
is found.

The matching is processed in the following way:
    - all edges within one intance of the original graph are added to the
    matching
    - for every edge going from a vertex to its copy the cheapes edge
    covering the vertex in the original graph is taken

"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def matching(cost_matrix):
    """Minumum cost maximum matching in a bipartite graph."""
    matching_x, matching_y = linear_sum_assignment(cost_matrix)
    return list(zip(matching_x, matching_y))


def edge_cover(cost_matrix, infinity=1e12):
    """Minimum weighted edge cover in a bipartite graph."""
    if cost_matrix.shape == (0, 0):
        return []

    x_dim, y_dim = cost_matrix.shape

    new_matrix = np.full([x_dim + y_dim, x_dim + y_dim], infinity)
    new_matrix[:x_dim, :y_dim] = cost_matrix
    new_matrix[x_dim:, y_dim:] = cost_matrix.T

    for i in range(x_dim):
        new_matrix[i][y_dim + i] = 2 * min(cost_matrix[i])

    for i in range(y_dim):
        new_matrix[x_dim + i][i] = 2 * min(cost_matrix[:, i])

    matching_x, matching_y = linear_sum_assignment(new_matrix)

    edge_cover_pairs = []
    for pair in zip(matching_x, matching_y):
        if pair[0] < x_dim and pair[1] < y_dim:
            edge_cover_pairs.append(pair)
        elif pair[0] == pair[1] - y_dim:
            target = np.argmin(cost_matrix[pair[0]])
            edge_cover_pairs.append((pair[0], target))
        elif pair[0] - x_dim == pair[1]:
            source = np.argmin(cost_matrix[:, pair[1]])
            edge_cover_pairs.append((source, pair[1]))

    return edge_cover_pairs


# def test():
#     matrices = [
#         #  Square
#         ([[400, 150, 400],
#           [400, 450, 600],
#           [300, 225, 300]], 850),
#         #  Rectangular variant
#         ([[400, 150, 400, 1],
#           [400, 450, 600, 2],
#           [300, 225, 300, 3]], 452),
#
#         #  Square
#         ([[10, 10, 8],
#           [9, 8, 1],
#           [9, 7, 4]], 18),
#
#         #  Rectangular variant
#         ([[10, 10, 8, 11],
#           [9, 8, 1, 1],
#           [9, 7, 4, 10]], 15),
#     ]
#
#     m = MinimumWeightedEdgeCover()
#     for cost_matrix, expected_total in matrices:
#         indexes = m.compute(cost_matrix)
#         total_cost = 0
#         for r, c in indexes:
#             x = cost_matrix[r][c]
#             total_cost += x
#             print('(%d, %d) -> %d' % (r, c, x))
#         print('lowest cost=%d' % total_cost)
#
#
# if __name__ == "__main__":
#    test()
