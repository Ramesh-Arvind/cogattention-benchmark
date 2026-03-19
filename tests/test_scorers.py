"""
Unit tests for all CogAttention scorers.
Tests correct scoring, edge cases, and adversarial response patterns.
"""

import unittest

from src.generators.capacity import generate_capacity_dataset
from src.generators.sustained import generate_sustained_dataset
from src.generators.selective import generate_selective_dataset
from src.generators.shifting import generate_shifting_dataset
from src.generators.anomaly import generate_anomaly_dataset
from src.generators.novel_capacity import generate_interference_dataset
from src.generators.novel_selective import generate_stroop_dataset
from src.generators.novel_sustained import generate_stream_dataset

from src.scorers.capacity import score_capacity, score_interference
from src.scorers.sustained import score_sustained, score_stream
from src.scorers.selective import score_selective, score_stroop
from src.scorers.shifting import score_shifting
from src.scorers.anomaly import score_anomaly
from src.scorers.composite import compute_cas
from src.scorers.base import extract_answer_block, extract_numbered_answers, extract_list_items


class TestBaseExtractors(unittest.TestCase):
    """Test response parsing utilities."""

    def test_extract_answer_block_standard(self):
        resp = "Some reasoning...\nANSWER:\n- Alice: red key"
        self.assertIn("Alice", extract_answer_block(resp))

    def test_extract_answer_block_lowercase(self):
        resp = "answer: 42"
        self.assertEqual(extract_answer_block(resp), "42")

    def test_extract_answer_block_no_marker(self):
        resp = "Just some text without a marker"
        self.assertEqual(extract_answer_block(resp), resp)

    def test_extract_numbered_answers(self):
        resp = "ANSWER:\n1. cat\n2. dog\n3. bird"
        answers = extract_numbered_answers(resp)
        self.assertEqual(answers["1"], "cat")
        self.assertEqual(answers["2"], "dog")
        self.assertEqual(answers["3"], "bird")

    def test_extract_numbered_with_colon(self):
        resp = "ANSWER:\n1: cat\n2: dog"
        answers = extract_numbered_answers(resp)
        self.assertEqual(answers["1"], "cat")

    def test_extract_list_comma_separated(self):
        resp = "ANSWER: eagle, heron, falcon"
        items = extract_list_items(resp)
        self.assertEqual(len(items), 3)
        self.assertIn("eagle", items)

    def test_extract_list_bullet_points(self):
        resp = "ANSWER:\n- eagle\n- heron\n- falcon"
        items = extract_list_items(resp)
        self.assertEqual(len(items), 3)

    def test_extract_list_numeric_with_commas(self):
        resp = "ANSWER: $3,201.33, $9,512.96, $6,474.86"
        items = extract_list_items(resp)
        self.assertGreaterEqual(len(items), 3)
        # Should not split on commas within numbers
        found_full = any("3,201" in item for item in items)
        self.assertTrue(found_full, f"Failed to keep comma in number: {items}")


class TestCapacityScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_capacity_dataset(seed=2026)

    def test_perfect_response(self):
        inst = self.dataset[0]
        gt = inst.gold_answer
        resp = "ANSWER:\n" + "\n".join(f"- {p}: {v}" for p, v in gt.items())
        result = score_capacity(inst, resp)
        self.assertEqual(result.metrics["accuracy"], 1.0)

    def test_empty_response(self):
        inst = self.dataset[0]
        result = score_capacity(inst, "")
        self.assertEqual(result.metrics["accuracy"], 0.0)

    def test_partial_response(self):
        inst = self.dataset[8]  # Medium difficulty, 3 people
        gt = inst.gold_answer
        people = list(gt.keys())
        # Only answer for first person correctly
        resp = f"ANSWER:\n- {people[0]}: {gt[people[0]]}"
        result = score_capacity(inst, resp)
        self.assertGreater(result.metrics["accuracy"], 0.0)
        self.assertLess(result.metrics["accuracy"], 1.0)

    def test_wrong_items(self):
        inst = self.dataset[0]
        people = inst.metadata["people"]
        resp = f"ANSWER:\n- {people[0]}: wrong item\n- {people[1]}: also wrong"
        result = score_capacity(inst, resp)
        self.assertEqual(result.metrics["accuracy"], 0.0)

    def test_response_with_preamble(self):
        """Model adds reasoning before the answer."""
        inst = self.dataset[0]
        gt = inst.gold_answer
        resp = ("Let me trace through the swaps step by step...\n"
                "After swap 1, Viktor has... After swap 2...\n\n"
                "ANSWER:\n" + "\n".join(f"- {p}: {v}" for p, v in gt.items()))
        result = score_capacity(inst, resp)
        self.assertEqual(result.metrics["accuracy"], 1.0)


class TestInterferenceScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_interference_dataset(seed=2026)

    def test_perfect_response(self):
        inst = self.dataset[0]
        gt = inst.gold_answer
        resp = "ANSWER:\n" + "\n".join(f"- {k}: {v}" for k, v in gt.items())
        result = score_interference(inst, resp)
        self.assertEqual(result.metrics["accuracy"], 1.0)

    def test_perseveration_detected(self):
        """Model gives an old value instead of the latest."""
        inst = self.dataset[0]
        key = inst.metadata["key_names"][0]
        prior = inst.metadata["all_prior_values"][key]
        if prior:
            old_val = prior[0]
            resp = f"ANSWER:\n- {key}: {old_val}"
            result = score_interference(inst, resp)
            self.assertGreater(result.metrics["perseveration_rate"], 0.0)


class TestSustainedScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_sustained_dataset(seed=2026)

    def test_perfect_response(self):
        inst = self.dataset[0]
        resp = "ANSWER: " + ", ".join(inst.gold_answer)
        result = score_sustained(inst, resp)
        self.assertEqual(result.metrics["recall"], 1.0)

    def test_partial_response(self):
        inst = self.dataset[0]
        resp = "ANSWER: " + inst.gold_answer[0]
        result = score_sustained(inst, resp)
        self.assertGreater(result.metrics["recall"], 0.0)
        self.assertLess(result.metrics["recall"], 1.0)

    def test_intrusion_with_nearmiss(self):
        inst = self.dataset[0]
        nm = inst.metadata["nearmisses"]
        resp = "ANSWER: " + ", ".join(inst.gold_answer + nm)
        result = score_sustained(inst, resp)
        self.assertEqual(result.metrics["recall"], 1.0)
        self.assertGreater(result.metrics["intrusion_rate"], 0.0)

    def test_empty_response(self):
        inst = self.dataset[0]
        result = score_sustained(inst, "I don't know")
        self.assertEqual(result.metrics["recall"], 0.0)


class TestSelectiveScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_selective_dataset(seed=2026)

    def test_perfect_selective(self):
        inst = self.dataset[0]
        resp = "ANSWER: " + ", ".join(inst.gold_answer)
        result = score_selective(inst, resp)
        self.assertGreater(result.metrics["sas_score"], 0.0)

    def test_all_distractors_included(self):
        """Model dumps everything — should have high intrusion."""
        inst = self.dataset[0]
        all_vals = inst.gold_answer + inst.metadata["distractor_values"]
        resp = "ANSWER: " + ", ".join(all_vals)
        result = score_selective(inst, resp)
        self.assertGreater(result.metrics["distractor_intrusion"], 0.0)


class TestStroopScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_stroop_dataset(seed=2026)

    def test_perfect_stroop(self):
        inst = self.dataset[0]
        resp = "ANSWER:\n" + "\n".join(f"{k}. {v}" for k, v in inst.gold_answer.items())
        result = score_stroop(inst, resp)
        self.assertEqual(result.metrics["accuracy"], 1.0)
        self.assertEqual(result.metrics["stroop_resistance"], 1.0)

    def test_stroop_error_detected(self):
        """Model corrects the error instead of answering literally."""
        inst = self.dataset[0]
        trap = inst.metadata["trap_answers"]
        # Use trap answers instead of correct ones
        resp = "ANSWER:\n" + "\n".join(
            f"{k}. {trap.get(k, 'unknown')}" for k in inst.gold_answer
        )
        result = score_stroop(inst, resp)
        if any(v for v in trap.values()):
            self.assertGreater(result.metrics["stroop_error_rate"], 0.0)


class TestShiftingScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_shifting_dataset(seed=2026)

    def test_perfect_shifting(self):
        inst = self.dataset[0]
        resp = "ANSWER:\n" + "\n".join(f"{k}. {v}" for k, v in inst.gold_answer.items())
        result = score_shifting(inst, resp)
        self.assertEqual(result.metrics["overall_accuracy"], 1.0)
        self.assertEqual(result.metrics["switch_cost"], 0.0)

    def test_perseveration_on_old_rule(self):
        """Model uses old rule for post-switch items."""
        inst = self.dataset[0]
        sp = inst.metadata["switch_point"]
        pre = inst.metadata["pre_answers"]
        persev = inst.metadata["perseveration_answers"]

        # Correct pre-switch, perseverated post-switch
        lines = []
        for i, (word, ans) in enumerate(pre, 1):
            lines.append(f"{i}. {ans}")
        for i, (word, ans) in enumerate(persev, sp + 1):
            lines.append(f"{i}. {ans}")

        resp = "ANSWER:\n" + "\n".join(lines)
        result = score_shifting(inst, resp)
        # Pre should be correct
        self.assertEqual(result.metrics["pre_switch_accuracy"], 1.0)
        # Post may show perseveration if rules produce different answers
        # (not guaranteed for every instance)

    def test_empty_response(self):
        inst = self.dataset[0]
        result = score_shifting(inst, "")
        self.assertEqual(result.metrics["overall_accuracy"], 0.0)


class TestAnomalyScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_anomaly_dataset(seed=2026)

    def test_perfect_both(self):
        inst = self.dataset[0]
        pa = inst.gold_answer["primary"]
        kw = inst.metadata["detection_keywords"][0]
        resp = f"ANSWER:\n1. {pa}\n2. Yes, I noticed {kw} in the passage."
        result = score_anomaly(inst, resp)
        self.assertEqual(result.metrics["primary_accuracy"], 1.0)
        self.assertEqual(result.metrics["anomaly_detection"], 1.0)
        self.assertEqual(result.metrics["dual_task_score"], 1.0)

    def test_primary_correct_anomaly_missed(self):
        inst = self.dataset[0]
        pa = inst.gold_answer["primary"]
        resp = f"ANSWER:\n1. {pa}\n2. nothing unusual"
        result = score_anomaly(inst, resp)
        self.assertEqual(result.metrics["primary_accuracy"], 1.0)
        self.assertEqual(result.metrics["anomaly_detection"], 0.0)
        # With weighted average: 0.4*1.0 + 0.6*0.0 = 0.4
        self.assertEqual(result.metrics["dual_task_score"], 0.4)

    def test_refusal_response(self):
        inst = self.dataset[0]
        resp = "I cannot complete this task."
        result = score_anomaly(inst, resp)
        self.assertEqual(result.metrics["primary_accuracy"], 0.0)


class TestCompositeScorer(unittest.TestCase):
    """Test CAS computation."""

    def test_cas_with_all_perfect(self):
        """CAS should be close to 1.0 with all perfect scores."""
        from src.scorers.base import ScoreResult
        results = []
        for task_type, metric in [
            ("capacity", "accuracy"), ("interference", "accuracy"),
            ("sustained", "recall"), ("stream_segregation", "stream_accuracy"),
            ("selective", "sas_score"), ("stroop", "accuracy"),
            ("shifting", "overall_accuracy"), ("anomaly", "dual_task_score"),
        ]:
            r = ScoreResult("test", task_type, "Easy")
            r.add_metric(metric, 1.0)
            results.append(r)

        cas = compute_cas(results)
        self.assertGreater(cas["cas_score"], 0.9)

    def test_cas_with_mixed(self):
        from src.scorers.base import ScoreResult
        results = []
        for task_type, metric, val in [
            ("capacity", "accuracy", 0.8),
            ("selective", "sas_score", 0.5),
            ("shifting", "overall_accuracy", 0.7),
        ]:
            r = ScoreResult("test", task_type, "Easy")
            r.add_metric(metric, val)
            results.append(r)

        cas = compute_cas(results)
        self.assertGreater(cas["cas_score"], 0.0)
        self.assertLess(cas["cas_score"], 1.0)


class TestNonCompensatoryScoring(unittest.TestCase):
    """Test geometric mean CAS (Step 2)."""

    def test_geometric_all_perfect(self):
        """All 1.0 scores → geometric = 1.0."""
        from src.scorers.base import ScoreResult
        results = []
        for task_type, metric in [
            ("capacity", "accuracy"), ("interference", "accuracy"),
            ("sustained", "recall"), ("stream_segregation", "stream_accuracy"),
            ("selective", "sas_score"), ("stroop", "accuracy"),
            ("shifting", "overall_accuracy"), ("anomaly", "dual_task_score"),
        ]:
            r = ScoreResult("test", task_type, "Easy")
            r.add_metric(metric, 1.0)
            results.append(r)

        cas = compute_cas(results)
        self.assertAlmostEqual(cas["cas_geometric"], 1.0, places=2)

    def test_geometric_one_zero(self):
        """One task at 0.0 → geometric ≈ 0.0 (epsilon floor)."""
        from src.scorers.base import ScoreResult
        results = []
        for task_type, metric, val in [
            ("capacity", "accuracy", 1.0),
            ("selective", "sas_score", 0.0),  # zero!
            ("shifting", "overall_accuracy", 1.0),
        ]:
            r = ScoreResult("test", task_type, "Easy")
            r.add_metric(metric, val)
            results.append(r)

        cas = compute_cas(results)
        self.assertLessEqual(cas["cas_geometric"], 0.01)

    def test_geometric_less_than_arithmetic(self):
        """Mixed scores → geometric ≤ arithmetic (AM-GM inequality)."""
        from src.scorers.base import ScoreResult
        results = []
        for task_type, metric, val in [
            ("capacity", "accuracy", 0.9),
            ("selective", "sas_score", 0.3),
            ("shifting", "overall_accuracy", 0.7),
        ]:
            r = ScoreResult("test", task_type, "Easy")
            r.add_metric(metric, val)
            results.append(r)

        cas = compute_cas(results)
        self.assertLessEqual(cas["cas_geometric"], cas["cas_score"] + 1e-6)

    def test_geometric_uniform_scores(self):
        """Identical scores → geometric == arithmetic."""
        from src.scorers.base import ScoreResult
        results = []
        for task_type, metric in [
            ("capacity", "accuracy"),
            ("selective", "sas_score"),
            ("shifting", "overall_accuracy"),
        ]:
            r = ScoreResult("test", task_type, "Easy")
            r.add_metric(metric, 0.6)
            results.append(r)

        cas = compute_cas(results)
        self.assertAlmostEqual(cas["cas_geometric"], cas["cas_score"], places=2)

    def test_backward_compatibility(self):
        """cas_score key still present alongside cas_geometric."""
        from src.scorers.base import ScoreResult
        r = ScoreResult("test", "capacity", "Easy")
        r.add_metric("accuracy", 0.5)
        cas = compute_cas([r])
        self.assertIn("cas_score", cas)
        self.assertIn("cas_geometric", cas)


class TestShiftingResidueClassification(unittest.TestCase):
    """Test attentional residue detection (Step 3)."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_shifting_dataset(seed=2026)

    def test_residue_detected(self):
        """Post-switch answer matching a pre-switch correct answer = residue."""
        inst = self.dataset[0]
        sp = inst.metadata["switch_point"]
        pre = inst.metadata["pre_answers"]
        post = inst.metadata["post_answers"]

        # Build response: correct pre-switch, then use a pre-switch answer post-switch
        lines = []
        for i, (word, ans) in enumerate(pre, 1):
            lines.append(f"{i}. {ans}")
        # For post-switch items, give an answer from pre_answers (not perseveration)
        pre_answer_val = pre[0][1]  # first pre-switch answer
        for i in range(sp + 1, sp + 1 + len(post)):
            lines.append(f"{i}. {pre_answer_val}")

        resp = "ANSWER:\n" + "\n".join(lines)
        result = score_shifting(inst, resp)

        # Should have residue or perseveration (depending on if pre answer matches old rule)
        total_residue = result.metrics["residue_error_count"]
        total_persev = result.metrics["perseveration_errors"]
        total_random = result.metrics["random_error_count"]
        # At least some post-switch errors should be classified
        post_errors = len(post) - int(result.metrics["post_switch_accuracy"] * len(post) + 0.5)
        if post_errors > 0:
            self.assertEqual(total_residue + total_persev + total_random, post_errors)

    def test_random_error_detected(self):
        """Completely unrelated answer → random error."""
        inst = self.dataset[0]
        sp = inst.metadata["switch_point"]
        pre = inst.metadata["pre_answers"]
        post = inst.metadata["post_answers"]

        lines = []
        for i, (word, ans) in enumerate(pre, 1):
            lines.append(f"{i}. {ans}")
        # Random gibberish for post-switch
        for i in range(sp + 1, sp + 1 + len(post)):
            lines.append(f"{i}. XYZZYSPOON")

        resp = "ANSWER:\n" + "\n".join(lines)
        result = score_shifting(inst, resp)
        self.assertGreater(result.metrics["random_error_count"], 0)
        self.assertEqual(result.metrics["residue_error_count"], 0)

    def test_backward_compatible_metrics(self):
        """Existing metrics unchanged — all original keys still present."""
        inst = self.dataset[0]
        resp = "ANSWER:\n" + "\n".join(f"{k}. {v}" for k, v in inst.gold_answer.items())
        result = score_shifting(inst, resp)

        for key in ["overall_accuracy", "pre_switch_accuracy", "post_switch_accuracy",
                     "switch_cost", "perseveration_rate", "perseveration_errors"]:
            self.assertIn(key, result.metrics, f"Missing key: {key}")

        # New keys also present
        for key in ["residue_error_count", "residue_rate", "random_error_count", "random_error_rate"]:
            self.assertIn(key, result.metrics, f"Missing new key: {key}")

    def test_all_correct_no_errors(self):
        """Perfect response → 0 residue, 0 random."""
        inst = self.dataset[0]
        resp = "ANSWER:\n" + "\n".join(f"{k}. {v}" for k, v in inst.gold_answer.items())
        result = score_shifting(inst, resp)
        self.assertEqual(result.metrics["overall_accuracy"], 1.0)
        self.assertEqual(result.metrics["residue_error_count"], 0)
        self.assertEqual(result.metrics["random_error_count"], 0)
        self.assertEqual(result.metrics["residue_rate"], 0.0)
        self.assertEqual(result.metrics["random_error_rate"], 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test adversarial LLM output patterns."""

    @classmethod
    def setUpClass(cls):
        cls.cap_inst = generate_capacity_dataset(seed=2026)[0]

    def test_markdown_formatting(self):
        """Model uses markdown bold/italic."""
        gt = self.cap_inst.gold_answer
        people = list(gt.keys())
        resp = "ANSWER:\n" + "\n".join(
            f"- **{p}**: *{gt[p]}*" for p in people
        )
        result = score_capacity(self.cap_inst, resp)
        # Should still parse (might not get perfect due to ** markers)
        # This tests robustness

    def test_extra_whitespace(self):
        gt = self.cap_inst.gold_answer
        people = list(gt.keys())
        resp = "ANSWER:  \n\n" + "\n\n".join(
            f"  -   {p}  :   {gt[p]}  " for p in people
        )
        result = score_capacity(self.cap_inst, resp)
        # Should handle gracefully

    def test_numbered_instead_of_bullets(self):
        gt = self.cap_inst.gold_answer
        people = list(gt.keys())
        resp = "ANSWER:\n" + "\n".join(
            f"{i+1}. {p}: {gt[p]}" for i, p in enumerate(people)
        )
        result = score_capacity(self.cap_inst, resp)
        # Person-item extraction should handle numbered format


if __name__ == "__main__":
    unittest.main(verbosity=2)
