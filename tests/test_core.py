import unittest

import numpy as np
import pandas as pd

from citynet.core import (
    build_city_graph,
    coupling_from_cost,
    great_circle_km,
    great_circle_matrix,
    graph_matrix,
    row_stochastic_matrix,
    symmetrize_matrix,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cities = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "name": ["A", "B", "C", "D"],
                "latitude": [0.0, 0.0, 1.0, 2.0],
                "longitude": [0.0, 1.0, 0.0, 0.0],
                "population": [10, 20, 30, 40],
            }
        )

    def test_haversine_one_degree(self):
        self.assertAlmostEqual(great_circle_km(0, 0, 1, 0), 111.195, places=2)

    def test_great_circle_matrix(self):
        matrix = great_circle_matrix(self.cities)
        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 0.0)

    def test_symmetrize(self):
        directed = np.array([[0, 2, np.inf], [4, 0, 6], [np.inf, 8, 0]], dtype=float)
        symmetric = symmetrize_matrix(directed, "mean")
        self.assertEqual(symmetric[0, 1], 3.0)
        self.assertTrue(np.isinf(symmetric[0, 2]))
        self.assertEqual(symmetric[1, 2], 7.0)

    def test_coupling(self):
        self.assertAlmostEqual(coupling_from_cost(0, model="exponential", scale=10), 1.0)
        self.assertLess(coupling_from_cost(20, model="exponential", scale=10), 1.0)
        self.assertGreater(
            coupling_from_cost(10, model="gravity", population_i=10, population_j=20),
            0.0,
        )

    def test_knn_graph_and_transition(self):
        costs = np.array(
            [
                [0, 1, 4, 9],
                [1, 0, 2, 8],
                [4, 2, 0, 3],
                [9, 8, 3, 0],
            ],
            dtype=float,
        )
        graph = build_city_graph(
            self.cities,
            selected_cost_matrix=costs,
            topology="knn",
            k=1,
            coupling_scale=5,
        )
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertGreaterEqual(graph.number_of_edges(), 2)
        coupling = graph_matrix(graph, node_order=["a", "b", "c", "d"], attribute="coupling")
        transition = row_stochastic_matrix(coupling)
        row_sums = transition.sum(axis=1)
        for value in row_sums:
            self.assertTrue(np.isclose(value, 0.0) or np.isclose(value, 1.0))

    def test_threshold_graph(self):
        costs = np.array(
            [
                [0, 1, 4, 9],
                [1, 0, 2, 8],
                [4, 2, 0, 3],
                [9, 8, 3, 0],
            ],
            dtype=float,
        )
        graph = build_city_graph(
            self.cities,
            selected_cost_matrix=costs,
            topology="threshold",
            threshold=2.5,
        )
        self.assertEqual(set(graph.edges()), {("a", "b"), ("b", "c")})


if __name__ == "__main__":
    unittest.main()
