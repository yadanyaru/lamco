import unittest

import numpy as np

from lamco import lamco_select, semantic_margin_scores


class LaMCoTests(unittest.TestCase):
    def test_margin_and_gate(self) -> None:
        prototypes = np.eye(2)
        images = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
        labels = np.asarray([0, 0, 0])
        margins, lcms, agrees = semantic_margin_scores(images, prototypes, labels)
        self.assertGreater(margins[0], 0)
        self.assertLess(margins[1], 0)
        self.assertEqual(agrees.tolist(), [True, False, True])
        self.assertEqual(lcms[1], 0.0)
        self.assertGreater(lcms[2], lcms[0])

    def test_exact_balanced_budget_and_determinism(self) -> None:
        rng = np.random.default_rng(3)
        prototypes = rng.normal(size=(3, 5))
        labels = np.repeat(np.arange(3), 6)
        images = prototypes[labels] + 0.3 * rng.normal(size=(18, 5))
        first = lamco_select(images, prototypes, labels, budget=8)
        second = lamco_select(images, prototypes, labels, budget=8)
        self.assertTrue(np.array_equal(first.indices, second.indices))
        self.assertEqual(len(np.unique(first.indices)), 8)
        self.assertEqual([item.budget for item in first.by_class.values()], [3, 3, 2])
        self.assertEqual(
            [int((labels[first.indices] == c).sum()) for c in range(3)], [3, 3, 2]
        )

    def test_boundary_branch_obeys_gate_when_consistent_points_exist(self) -> None:
        prototypes = np.eye(2)
        images = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
        labels = np.asarray([0, 0, 0, 1])
        result = lamco_select(images, prototypes, labels, budget=3, rho=1.0)
        boundary_zero = result.by_class[0].boundary
        self.assertTrue(all(result.agrees_with_label[boundary_zero]))

    def test_requires_one_budget_form(self) -> None:
        with self.assertRaises(ValueError):
            lamco_select(np.eye(2), np.eye(2), np.arange(2))


if __name__ == "__main__":
    unittest.main()
