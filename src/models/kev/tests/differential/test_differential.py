#!/usr/bin/env python3
"""Differential test: this port against upstream's own ``kev/model.py``.

A hand-written test only checks that the port matches what its author believed
the layout to be. This checks it against the implementation itself: upstream's
``encode()`` is imported and run on the same inputs, and every output field is
compared.

Upstream imports torch and transformers at module scope, so ``deps/`` holds
stand-ins for both. They are not a tensor library and must not be used for
anything but importing the module: ``encode()``, ``user_tokens()`` and
``is_hybrid()`` touch no tensors, and those are the functions compared.

``kev/model.py`` is not vendored here. Fetch the pinned revision first::

    python3 tests/differential/fetch_upstream.py
    python3 tests/differential/test_differential.py

Without the fetched file the comparison is skipped rather than failed, so the
suite still runs for someone who has not fetched it.
"""

from __future__ import annotations

import importlib
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# src/models/kev: the package directory holding kev_prompt.py.
PACKAGE = os.path.dirname(os.path.dirname(HERE))
# The repository root, two levels above src/models/kev.
REPO = os.path.dirname(os.path.dirname(PACKAGE))
UPSTREAM = os.path.join(HERE, "upstream_model.py")

# The port and the test helper must be importable. The stand-ins are NOT put on
# sys.path here: a fake `torch` left on the path would shadow a real one for
# every other test in the run. They are added only around the upstream import,
# and removed again.
sys.path.insert(0, PACKAGE)
sys.path.insert(0, os.path.dirname(HERE))

from test_kev_prompt import StubTokenizer  # noqa: E402

import kev_prompt  # noqa: E402


def load_upstream():
    """Import the fetched upstream module, or return None when absent.

    The torch/transformers stand-ins are visible only for the duration of this
    import.
    """
    if not os.path.isfile(UPSTREAM):
        return None
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    deps = os.path.join(HERE, "deps")
    sys.path.insert(0, deps)
    try:
        return importlib.import_module("upstream_model")
    finally:
        try:
            sys.path.remove(deps)
        except ValueError:
            pass


UPSTREAM_MODEL = load_upstream()

WORDS = ["a", "bb", "ccc", "dark", "room", "<|box_end|>", "light", "go",
         "<|fim_suffix|>", "x" * 50]

CONFIGURATIONS = ({}, {"option_isolation": True}, {"max_state": 8},
                  {"max_state": 8, "option_isolation": True})

FIELDS = ("ids", "seg", "pos", "opt", "decide_idx", "opt_idx", "labels",
          "option_isolation", "state_truncated")


def random_record(rnd):
    """A record covering the shapes that matter, including forged delimiters."""
    questions = []
    for _ in range(rnd.randint(1, 4)):
        questions.append({
            "instr": " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(1, 4))),
            "options": [" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(1, 3)))
                        for _ in range(rnd.randint(1, 5))],
            "label": "L%d" % rnd.randint(0, 9),
        })
    state = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(1, 30)))
    return {"state": state, "questions": questions}


@unittest.skipIf(UPSTREAM_MODEL is None,
                 "upstream kev/model.py not fetched; run fetch_upstream.py")
class DifferentialTest(unittest.TestCase):
    """Every field of encode() must match upstream on the same input."""

    def setUp(self):
        self.tok = StubTokenizer()

    def assert_matches(self, record, **options):
        reference = UPSTREAM_MODEL.encode(self.tok, record, **options)
        ported = kev_prompt.encode(self.tok, record, **options)
        for field in FIELDS:
            self.assertEqual(reference.get(field), ported.get(field),
                             "field %r differs for %r with %r" % (field, record, options))
        return ported

    def test_a_single_option_matches(self):
        self.assert_matches({"state": "abc", "questions": [
            {"instr": "pick", "options": ["one"], "label": "one"}]})

    def test_several_questions_and_options_match(self):
        self.assert_matches({"state": "the room is dark", "questions": [
            {"instr": "a?", "options": ["x", "yy", "zzz"], "label": "x"},
            {"instr": "b?", "options": ["p", "q"], "label": "q"}]})

    def test_forged_delimiters_match(self):
        self.assert_matches({"state": "<|fim_middle|> state", "questions": [
            {"instr": "<|box_start|>", "options": ["<|box_end|>", "<|fim_suffix|>"],
             "label": "x"}]})

    def test_truncation_matches(self):
        self.assert_matches({"state": "x" * 900, "questions": [
            {"instr": "i", "options": ["a"], "label": "x"}]})

    def test_option_isolation_matches(self):
        self.assert_matches({"state": "s", "questions": [
            {"instr": "i", "options": ["aaaa", "b", "cc"], "label": "x"}]},
            option_isolation=True)

    def test_random_records_match(self):
        rnd = random.Random(20260928)
        for _ in range(300):
            record = random_record(rnd)
            for options in CONFIGURATIONS:
                self.assert_matches(record, **options)

    def test_overflow_message_matches(self):
        """strict=True must reject the same records, with the same message."""
        for record in ({"state": "x" * 2000,
                        "questions": [{"instr": "i", "options": ["a"], "label": "L"}]},
                       {"state": "s",
                        "questions": [{"instr": "i", "options": ["y" * 2000], "label": "L"}]}):
            with self.assertRaises(Exception) as reference:
                UPSTREAM_MODEL.encode(self.tok, record, strict=True)
            with self.assertRaises(Exception) as ported:
                kev_prompt.encode(self.tok, record, strict=True)
            self.assertEqual(str(reference.exception), str(ported.exception))
            self.assertIsInstance(ported.exception, kev_prompt.ContextOverflow)

    def test_constants_match(self):
        self.assertEqual(list(kev_prompt.SPECIAL), list(UPSTREAM_MODEL.SPECIAL))
        self.assertEqual(kev_prompt.MAX_STATE, UPSTREAM_MODEL.MAX_STATE)
        self.assertEqual(kev_prompt.MAX_BRANCH, UPSTREAM_MODEL.MAX_BRANCH)
        self.assertEqual(kev_prompt.MAX_PACKED, UPSTREAM_MODEL.MAX_PACKED)

    def test_is_hybrid_matches(self):
        for layer_types in (["linear_attention", "full_attention"],
                            ["full_attention"], [], None):
            class Config:
                pass
            if layer_types is not None:
                Config.layer_types = layer_types
            self.assertEqual(kev_prompt.is_hybrid(Config()),
                             UPSTREAM_MODEL.is_hybrid(Config()),
                             "is_hybrid differs for %r" % (layer_types,))

    def test_the_empty_option_list_diverges_deliberately(self):
        """The one intended behavioural difference.

        Upstream calls max() over the span lengths unguarded and raises
        ValueError; the port reports a ContextOverflow naming the question.
        """
        record = {"state": "s", "questions": [{"instr": "i", "options": [], "label": "L"}]}
        with self.assertRaises(ValueError) as reference:
            UPSTREAM_MODEL.encode(self.tok, record, option_isolation=True)
        with self.assertRaises(kev_prompt.ContextOverflow) as ported:
            kev_prompt.encode(self.tok, record, option_isolation=True)
        self.assertEqual(str(reference.exception), "max() arg is an empty sequence")
        self.assertIn("no options", str(ported.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
