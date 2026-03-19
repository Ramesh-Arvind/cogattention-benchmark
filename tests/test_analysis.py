"""
Unit tests for analysis modules: degradation, bootstrap, position_bias.
"""

import unittest
import math
import numpy as np

from src.scorers.base import ScoreResult
from src.analysis.degradation import compute_degradation_coefficient
from src.analysis.bootstrap import bootstrap_ci, bootstrap_cas_ci
from src.analysis.position_bias import compute_position_bias


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_result(task_type, difficulty="Easy", **metrics):
    r = ScoreResult("test", task_type, difficulty)
    for k, v in metrics.items():
        r.add_metric(k, v)
    return r


# ── Step 4: Degradation Coefficient ─────────────────────────────────────────

class TestDegradationCoefficient(unittest.TestCase):

    def test_degradation_perfect_power_law(self):
        """Synthetic data with known delta should be recovered."""
        # error_rate = 0.5 * noise_ratio^2.0
        known_delta = 2.0
        known_a = 0.5
        results = []
        for nr in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            error = known_a * (nr ** known_delta)
            score = 1.0 - error
            r = _make_result("selective", sas_score=score, noise_ratio=nr)
            results.append(r)

        out = compute_degradation_coefficient(results, task_type="selective")
        self.assertAlmostEqual(out["delta"], known_delta, places=1)
        self.assertGreater(out["r_squared"], 0.99)
        self.assertAlmostEqual(out["a"], known_a, places=1)

    def test_degradation_all_correct(self):
        """When all scores are 1.0, error is 0 — should handle gracefully."""
        results = [
            _make_result("selective", sas_score=1.0, noise_ratio=0.3),
            _make_result("selective", sas_score=1.0, noise_ratio=0.5),
            _make_result("selective", sas_score=1.0, noise_ratio=0.7),
        ]
        out = compute_degradation_coefficient(results, task_type="selective")
        self.assertEqual(out["delta"], 0.0)
        self.assertEqual(out["n_points"], 0)  # all bins have zero error

    def test_degradation_monotonic(self):
        """Positive delta for increasing error with noise."""
        results = []
        for nr, score in [(0.2, 0.95), (0.4, 0.80), (0.6, 0.60), (0.8, 0.35)]:
            r = _make_result("selective", sas_score=score, noise_ratio=nr)
            results.append(r)

        out = compute_degradation_coefficient(results, task_type="selective")
        self.assertGreater(out["delta"], 0.0)

    def test_noise_ratio_in_selective_metrics(self):
        """Verify noise_ratio appears in ScoreResult metrics from selective scorer."""
        from src.generators.selective import generate_selective_dataset
        from src.scorers.selective import score_selective

        dataset = generate_selective_dataset(seed=9999)
        inst = dataset[0]
        resp = "ANSWER: " + ", ".join(inst.gold_answer)
        result = score_selective(inst, resp)
        self.assertIn("noise_ratio", result.metrics)

    def test_empty_results(self):
        """Empty input should return zeros."""
        out = compute_degradation_coefficient([], task_type="selective")
        self.assertEqual(out["delta"], 0.0)
        self.assertEqual(out["n_points"], 0)


# ── Step 5: Bootstrap Confidence Intervals ───────────────────────────────────

class TestBootstrapCI(unittest.TestCase):

    def test_ci_all_same_values(self):
        """CI width should be 0 when all values are the same."""
        ci = bootstrap_ci([0.5] * 100, n_bootstrap=5000, seed=42)
        self.assertAlmostEqual(ci["ci_lower"], ci["ci_upper"], places=4)
        self.assertAlmostEqual(ci["point_estimate"], 0.5, places=4)

    def test_ci_contains_true_mean(self):
        """For a known distribution, the true mean should be within the CI."""
        rng = np.random.RandomState(123)
        values = list(rng.normal(0.7, 0.1, size=200))
        ci = bootstrap_ci(values, n_bootstrap=10000, seed=42)
        self.assertLessEqual(ci["ci_lower"], 0.7)
        self.assertGreaterEqual(ci["ci_upper"], 0.7)

    def test_ci_wider_with_variance(self):
        """Higher variance should produce wider CI."""
        rng = np.random.RandomState(42)
        low_var = list(rng.normal(0.5, 0.01, size=50))
        high_var = list(rng.normal(0.5, 0.3, size=50))

        ci_low = bootstrap_ci(low_var, n_bootstrap=5000, seed=42)
        ci_high = bootstrap_ci(high_var, n_bootstrap=5000, seed=42)

        width_low = ci_low["ci_upper"] - ci_low["ci_lower"]
        width_high = ci_high["ci_upper"] - ci_high["ci_lower"]
        self.assertGreater(width_high, width_low)

    def test_ci_respects_level(self):
        """99% CI should be wider than 95% CI."""
        rng = np.random.RandomState(42)
        values = list(rng.normal(0.5, 0.1, size=100))

        ci_95 = bootstrap_ci(values, n_bootstrap=5000, ci_level=0.95, seed=42)
        ci_99 = bootstrap_ci(values, n_bootstrap=5000, ci_level=0.99, seed=42)

        width_95 = ci_95["ci_upper"] - ci_95["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        self.assertGreater(width_99, width_95)

    def test_bootstrap_cas_ci_structure(self):
        """Output should have 'cas' key and per-task keys."""
        results = []
        for tt in ["capacity", "sustained", "selective"]:
            for _ in range(5):
                r = _make_result(tt, accuracy=0.8, recall=0.8, sas_score=0.8)
                results.append(r)

        out = bootstrap_cas_ci(results, n_bootstrap=500, seed=42)
        self.assertIn("cas", out)
        self.assertIn("point_estimate", out["cas"])
        self.assertIn("ci_lower", out["cas"])
        self.assertIn("ci_upper", out["cas"])
        # Per-task keys
        for tt in ["capacity", "sustained", "selective"]:
            self.assertIn(tt, out)

    def test_empty_values(self):
        """Empty input should return zeros."""
        ci = bootstrap_ci([])
        self.assertEqual(ci["point_estimate"], 0.0)

    def test_single_value(self):
        """Single value: CI should collapse to that value."""
        ci = bootstrap_ci([0.42])
        self.assertAlmostEqual(ci["point_estimate"], 0.42, places=2)
        self.assertAlmostEqual(ci["ci_lower"], 0.42, places=2)


# ── Step 6: Position Bias ────────────────────────────────────────────────────

class TestPositionBias(unittest.TestCase):

    def test_position_bias_uniform(self):
        """No U-shape when accuracy is constant across positions."""
        results = []
        for pos in np.linspace(0.0, 1.0, 50):
            r = _make_result("sustained", accuracy=0.8)
            r.metrics["target_position_ratio"] = round(float(pos), 2)
            results.append(r)

        out = compute_position_bias(results)
        self.assertFalse(out["has_u_shape"])

    def test_position_bias_structure(self):
        """Output should have correct keys."""
        results = []
        for pos in np.linspace(0.0, 0.99, 20):
            r = _make_result("sustained", accuracy=0.7)
            r.metrics["target_position_ratio"] = round(float(pos), 2)
            results.append(r)

        out = compute_position_bias(results)
        self.assertIn("bins", out)
        self.assertIn("has_u_shape", out)
        self.assertEqual(len(out["bins"]), 5)  # default n_bins=5
        for b in out["bins"]:
            self.assertIn("bin_label", b)
            self.assertIn("accuracy_mean", b)
            self.assertIn("n", b)

    def test_position_bias_u_shape_detected(self):
        """Should detect U-shape when edges are higher than middle."""
        results = []
        # U-shape: high at start and end, low in middle
        for pos, acc in [(0.05, 0.9), (0.15, 0.9),
                         (0.25, 0.5), (0.35, 0.5),
                         (0.45, 0.4), (0.55, 0.4),
                         (0.65, 0.5), (0.75, 0.5),
                         (0.85, 0.9), (0.95, 0.9)]:
            r = _make_result("sustained", accuracy=acc)
            r.metrics["target_position_ratio"] = pos
            results.append(r)

        out = compute_position_bias(results)
        self.assertTrue(out["has_u_shape"])

    def test_empty_results(self):
        """Empty input should return empty bins."""
        out = compute_position_bias([])
        self.assertEqual(len(out["bins"]), 0)
        self.assertFalse(out["has_u_shape"])


# ── Effect Size ───────────────────────────────────────────────────────────────

class TestEffectSize(unittest.TestCase):

    def test_cohens_d_identical(self):
        """Identical groups → d = 0."""
        from src.analysis.effect_size import cohens_d
        d = cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(d, 0.0, places=4)

    def test_cohens_d_large_effect(self):
        """Well-separated groups → large d."""
        from src.analysis.effect_size import cohens_d, interpret_cohens_d
        d = cohens_d([0.9, 0.95, 0.85, 0.9], [0.2, 0.25, 0.15, 0.2])
        self.assertGreater(abs(d), 0.8)
        self.assertEqual(interpret_cohens_d(d), "large")

    def test_rank_biserial_perfect(self):
        """All A > all B → rbc = 1.0."""
        from src.analysis.effect_size import rank_biserial
        rbc = rank_biserial([0.9, 0.8, 0.7], [0.3, 0.2, 0.1])
        self.assertAlmostEqual(rbc, 1.0, places=4)

    def test_rank_biserial_equal(self):
        """Identical groups → rbc = 0."""
        from src.analysis.effect_size import rank_biserial
        rbc = rank_biserial([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(rbc, 0.0, places=4)

    def test_pairwise_effects_structure(self):
        """Output should have correct structure."""
        from src.analysis.effect_size import compute_pairwise_effects
        results_a = [_make_result("capacity", accuracy=0.9) for _ in range(5)]
        results_b = [_make_result("capacity", accuracy=0.5) for _ in range(5)]
        out = compute_pairwise_effects({"ModelA": results_a, "ModelB": results_b})
        self.assertEqual(len(out), 1)
        key = list(out.keys())[0]
        self.assertIn("overall_cohens_d", out[key])
        self.assertIn("per_task", out[key])
        self.assertIn("capacity", out[key]["per_task"])

    def test_interpret_thresholds(self):
        from src.analysis.effect_size import interpret_cohens_d
        self.assertEqual(interpret_cohens_d(0.1), "negligible")
        self.assertEqual(interpret_cohens_d(0.3), "small")
        self.assertEqual(interpret_cohens_d(0.6), "medium")
        self.assertEqual(interpret_cohens_d(1.2), "large")


# ── IRT Analysis ─────────────────────────────────────────────────────────────

class TestIRT(unittest.TestCase):

    def test_2pl_fit_basic(self):
        """2PL should fit and return expected keys."""
        from src.analysis.irt import fit_irt_model
        # 3 models, 5 items — model 0 is strongest
        response_matrix = np.array([
            [1.0, 1.0, 0.8, 0.6, 0.3],  # strong
            [0.8, 0.6, 0.5, 0.3, 0.1],  # medium
            [0.5, 0.3, 0.2, 0.1, 0.0],  # weak
        ])
        out = fit_irt_model(response_matrix, ["Strong", "Medium", "Weak"])
        self.assertIn("theta", out)
        self.assertIn("difficulty", out)
        self.assertIn("discrimination", out)
        self.assertEqual(len(out["theta"]), 3)
        self.assertEqual(len(out["difficulty"]), 5)
        # Strong model should have highest theta
        self.assertGreater(out["theta"]["Strong"], out["theta"]["Weak"])

    def test_1pl_fit(self):
        """1PL should fix all discriminations near 1.0."""
        from src.analysis.irt import fit_irt_model
        response_matrix = np.array([
            [1.0, 0.8, 0.5],
            [0.7, 0.5, 0.2],
        ])
        out = fit_irt_model(response_matrix, model_type="1PL")
        for a in out["discrimination"].values():
            self.assertAlmostEqual(a, 1.0, places=1)

    def test_build_response_matrix(self):
        """Response matrix should have correct shape."""
        from src.analysis.irt import build_response_matrix
        results_a = []
        results_b = []
        for i in range(5):
            r = ScoreResult(f"item_{i}", "capacity", "Easy")
            r.add_metric("accuracy", 0.9)
            results_a.append(r)
            r2 = ScoreResult(f"item_{i}", "capacity", "Easy")
            r2.add_metric("accuracy", 0.5)
            results_b.append(r2)

        matrix, models, items = build_response_matrix({"A": results_a, "B": results_b})
        self.assertEqual(matrix.shape, (2, 5))
        self.assertEqual(len(models), 2)
        self.assertEqual(len(items), 5)

    def test_difficulty_ordering(self):
        """Easy items should have lower difficulty than hard items."""
        from src.analysis.irt import fit_irt_model
        # Item 0 is easy (all models get it), Item 4 is hard (only best model)
        response_matrix = np.array([
            [1.0, 1.0, 0.9, 0.7, 0.4],
            [1.0, 0.9, 0.6, 0.3, 0.1],
            [0.9, 0.7, 0.3, 0.1, 0.0],
        ])
        out = fit_irt_model(response_matrix)
        # Item_0 should have lower difficulty than Item_4
        self.assertLess(out["difficulty"]["Item_0"], out["difficulty"]["Item_4"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
