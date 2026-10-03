#!/usr/bin/env python3
"""Tests for the pointer head and its readout.

Synthetic weights make the invariants deterministic; the real ``head.pt`` is
loaded too when the research snapshot is present, because the shipped checkpoint
is the one whose ids and shapes have to be right.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kev_head  # noqa: E402

# The checkpoint is 5 MB and belongs to someone else's repository, so it is not
# committed and the tests that need it skip. Point KEV_HEAD_PT at a copy to run
# them; tests/differential/fetch_upstream.py shows the pattern for fetching.
CHECKPOINT = os.environ.get("KEV_HEAD_PT", "")

# Kev-4B: hidden 2560, pointer dim 256.
D, DP = 2560, 256


def synthetic(seed=0, temperature=1.0):
    rng = np.random.default_rng(seed)
    return kev_head.PointerHead(
        q_weight=rng.standard_normal((DP, D), dtype=np.float32) * 0.02,
        q_bias=rng.standard_normal(DP, dtype=np.float32) * 0.02,
        k_weight=rng.standard_normal((DP, D), dtype=np.float32) * 0.02,
        k_bias=rng.standard_normal(DP, dtype=np.float32) * 0.02,
        temperature=temperature,
    )


class ShapeTest(unittest.TestCase):
    def test_scale_is_one_over_root_pointer_dim(self):
        head = synthetic()
        self.assertEqual(head.head_dim, DP)
        self.assertAlmostEqual(head.scale, 1.0 / np.sqrt(DP), places=12)

    def test_parameter_count(self):
        # 2 * (256*2560 + 256)
        self.assertEqual(synthetic().parameters, 1_311_232)

    def test_mismatched_q_and_k_are_rejected(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            kev_head.PointerHead(
                q_weight=rng.standard_normal((DP, D), dtype=np.float32),
                q_bias=rng.standard_normal(DP, dtype=np.float32),
                k_weight=rng.standard_normal((DP + 1, D), dtype=np.float32),
                k_bias=rng.standard_normal(DP + 1, dtype=np.float32))


class ReadoutTest(unittest.TestCase):
    def setUp(self):
        self.head = synthetic(seed=1)
        self.rng = np.random.default_rng(2)
        self.decide = self.rng.standard_normal(D, dtype=np.float32)
        self.options = self.rng.standard_normal((4, D), dtype=np.float32)

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(float(self.head.probs(self.decide, self.options).sum()),
                               1.0, places=10)

    def test_probabilities_are_bounded_and_ordered_like_the_logits(self):
        logits = self.head.logits(self.decide, self.options)
        probs = self.head.probs(self.decide, self.options)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertEqual(int(np.argmax(logits)), int(np.argmax(probs)))

    def test_identical_options_get_identical_probability(self):
        same = np.vstack([self.options[0]] * 3)
        probs = self.head.probs(self.decide, same)
        for value in probs:
            self.assertAlmostEqual(float(value), 1.0 / 3.0, places=10)

    def test_a_single_option_gets_all_the_probability(self):
        probs = self.head.probs(self.decide, self.options[:1])
        self.assertAlmostEqual(float(probs[0]), 1.0, places=10)

    def test_larger_activations_separate_the_candidates_more(self):
        small = self.head.probs(self.decide * 0.1, self.options * 0.1)
        large = self.head.probs(self.decide * 3.0, self.options * 3.0)
        self.assertGreater(float(large.max()), float(small.max()))


class TemperatureTest(unittest.TestCase):
    def test_temperature_never_changes_the_argmax(self):
        # Upstream notes the argmax is unchanged by construction.
        head = synthetic(seed=3, temperature=2.406050072164233)
        rng = np.random.default_rng(4)
        for _ in range(200):
            decide = rng.standard_normal(D, dtype=np.float32)
            options = rng.standard_normal((5, D), dtype=np.float32)
            hot = head.logits(decide, options, apply_temperature=True)
            raw = head.logits(decide, options, apply_temperature=False)
            self.assertEqual(int(hot.argmax()), int(raw.argmax()))

    def test_temperature_flattens_the_distribution(self):
        head = synthetic(seed=5, temperature=2.406050072164233)
        rng = np.random.default_rng(6)
        decide = rng.standard_normal(D, dtype=np.float32) * 3
        options = rng.standard_normal((4, D), dtype=np.float32) * 3
        hot = head.probs(decide, options, apply_temperature=True)
        raw = head.probs(decide, options, apply_temperature=False)
        self.assertLess(float(hot.max()), float(raw.max()))

    def test_temperature_of_one_is_a_no_op(self):
        head = synthetic(seed=7, temperature=1.0)
        rng = np.random.default_rng(8)
        decide = rng.standard_normal(D, dtype=np.float32)
        options = rng.standard_normal((3, D), dtype=np.float32)
        np.testing.assert_allclose(head.logits(decide, options, True),
                                   head.logits(decide, options, False))

    def test_probs_can_be_taken_without_temperature(self):
        head = synthetic(seed=9, temperature=2.4)
        rng = np.random.default_rng(10)
        decide = rng.standard_normal(D, dtype=np.float32)
        options = rng.standard_normal((3, D), dtype=np.float32)
        self.assertAlmostEqual(
            float(head.probs(decide, options, apply_temperature=False).sum()), 1.0, places=10)


class BatchedTest(unittest.TestCase):
    def setUp(self):
        self.head = synthetic(seed=11)
        self.rng = np.random.default_rng(12)
        self.counts = [2, 3, 4]
        self.decides = self.rng.standard_normal((3, D), dtype=np.float32)
        self.options = self.rng.standard_normal((sum(self.counts), D), dtype=np.float32)
        self.owner = np.repeat(np.arange(3), self.counts)

    def test_many_matches_one_at_a_time(self):
        batched = self.head.logits_many(self.decides, self.options, self.owner)
        for question in range(3):
            single = self.head.logits(self.decides[question],
                                      self.options[self.owner == question])
            np.testing.assert_allclose(batched[self.owner == question], single, atol=1e-5)

    def test_probabilities_are_taken_within_each_question(self):
        per_question = self.head.probs_by_question(self.decides, self.options, self.owner)
        self.assertEqual(len(per_question), 3)
        for question, probs in enumerate(per_question):
            self.assertEqual(len(probs), self.counts[question])
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)

    def test_owner_length_must_match_the_option_rows(self):
        with self.assertRaises(ValueError):
            self.head.logits_many(self.decides, self.options, self.owner[:-1])


@unittest.skipUnless(CHECKPOINT and os.path.isfile(CHECKPOINT),
                     "set KEV_HEAD_PT to a head.pt copy to run these")
class RealCheckpointTest(unittest.TestCase):
    """The shipped checkpoint is the one whose shapes and ids have to be right."""

    @classmethod
    def setUpClass(cls):
        cls.head = kev_head.load_head(CHECKPOINT)

    def test_shape_and_parameter_count(self):
        self.assertEqual(self.head.q_weight.shape, (DP, D))
        self.assertEqual(self.head.k_weight.shape, (DP, D))
        self.assertEqual(self.head.q_bias.shape, (DP,))
        self.assertEqual(self.head.parameters, 1_311_232)

    def test_temperature_is_the_fitted_value(self):
        self.assertAlmostEqual(self.head.temperature, 2.406050072164233, places=12)

    def test_head_is_float32(self):
        self.assertEqual(self.head.q_weight.dtype, np.float32)
        self.assertEqual(self.head.k_weight.dtype, np.float32)

    def test_a_question_with_four_candidates_is_a_distribution(self):
        rng = np.random.default_rng(13)
        probs = self.head.probs(rng.standard_normal(D, dtype=np.float32),
                                rng.standard_normal((4, D), dtype=np.float32))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)

    def test_the_trained_head_separates_candidates_at_realistic_scale(self):
        # At unit-scale activations the logits must actually spread, otherwise
        # every answer would be near-uniform and the model would be useless.
        rng = np.random.default_rng(14)
        logits = self.head.logits(rng.standard_normal(D, dtype=np.float32),
                                  rng.standard_normal((4, D), dtype=np.float32))
        self.assertGreater(float(logits.max() - logits.min()), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
