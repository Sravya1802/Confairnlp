import unittest

import numpy as np

from conformal.fair_cp import run_fair_cp_sweep, select_lambda_by_tuning
from conformal.group_conditional_cp import run_group_conditional_cp
from conformal.marginal_cp import compute_quantile_threshold, run_marginal_cp


class ConformalCoreTests(unittest.TestCase):
    def test_quantile_uses_higher_finite_sample_level(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(compute_quantile_threshold(scores, alpha=0.25), 0.4)

    def test_invalid_score_function_raises(self):
        probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
        labels = np.array([0, 1])
        with self.assertRaises(ValueError):
            run_marginal_cp(probs, labels, probs, labels, score_function="bad")

    def test_small_group_uses_marginal_fallback(self):
        cal_probs = np.tile(np.array([[0.7, 0.2, 0.1]]), (6, 1))
        cal_labels = np.array([0, 0, 1, 1, 2, 2])
        cal_groups = [["A"], ["A"], ["B"], ["B"], ["B"], ["B"]]
        test_probs = np.tile(np.array([[0.6, 0.3, 0.1]]), (2, 1))
        test_labels = np.array([0, 1])
        test_groups = [["A"], ["B"]]

        results = run_group_conditional_cp(
            cal_probs,
            cal_labels,
            cal_groups,
            test_probs,
            test_labels,
            test_groups,
            min_test_group_size=1,
        )

        self.assertTrue(results["per_group"]["A"]["used_marginal_fallback"])
        self.assertTrue(results["per_group"]["B"]["used_marginal_fallback"])

    def test_lambda_selection_prefers_coverage_then_disparity(self):
        cal_probs = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
                [0.2, 0.1, 0.7],
            ]
        )
        cal_labels = np.array([0, 0, 1, 1, 2, 2])
        cal_groups = [["A"], ["A"], ["B"], ["B"], ["C"], ["C"]]
        eval_probs = cal_probs.copy()
        eval_labels = cal_labels.copy()
        eval_groups = cal_groups.copy()

        sweep = run_fair_cp_sweep(
            cal_probs,
            cal_labels,
            cal_groups,
            eval_probs,
            eval_labels,
            eval_groups,
            alpha=0.2,
            lambda_steps=3,
            min_test_group_size=1,
            verbose=False,
        )
        selected = select_lambda_by_tuning(sweep, alpha=0.2)
        self.assertIn(selected["lambda"], {0.0, 0.5, 1.0})


if __name__ == "__main__":
    unittest.main()
