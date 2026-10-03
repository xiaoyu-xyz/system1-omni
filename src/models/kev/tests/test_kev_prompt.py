#!/usr/bin/env python3
"""Tests for the Kev prompt builder.

No GPU, no checkpoint, no PyTorch. The token ids are the real ones from
``jaredpalmer/kev-4b``'s ``added_tokens.json``, and the stub tokenizer reproduces
the one property the design depends on: a delimiter written by a caller is
tokenized as ordinary text, never as the delimiter it resembles.

The port is checked against the layout in ``kev/model.py``: the order of
delimiters, which token the head reads for each option, that branch positions
restart after the state, and that forged delimiters cannot create options.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kev_prompt  # noqa: E402  (path is set above)


class Batch(dict):
    """Stands in for a tokenizer's BatchEncoding."""

    @property
    def input_ids(self):
        return self["input_ids"]


class StubTokenizer:
    """Real special-token ids; ordinary text becomes one id per character.

    A delimiter-shaped string written by a caller is tokenized as text. That is
    the property ``user_tokens()`` relies on, and reproducing it here is what
    lets a test show that forged delimiters do not become structure.
    """

    def __init__(self):
        self.special = dict(zip(kev_prompt.SPECIAL, kev_prompt.SPECIAL_IDS))
        self._text_ids = {}

    def convert_tokens_to_ids(self, token):
        if token in self.special:
            return self.special[token]
        return self._text_ids.setdefault(token, 200000 + len(self._text_ids))

    def _char(self, char):
        return 1000 + ord(char) % 1000

    def __call__(self, text, add_special_tokens=False):
        ids = []
        index = 0
        # Longest-match the real delimiters, exactly as a fast tokenizer would.
        tokens = sorted(self.special, key=len, reverse=True)
        while index < len(text):
            for token in tokens:
                if text.startswith(token, index):
                    ids.append(self.special[token])
                    index += len(token)
                    break
            else:
                ids.append(self._char(text[index]))
                index += 1
        return Batch(input_ids=ids)


def record(state="the room is dark", questions=None):
    if questions is None:
        questions = [{"instr": "which action?", "options": ["turn on light", "leave"],
                      "label": "turn on light"}]
    return {"state": state, "questions": questions}


class ConstantsTest(unittest.TestCase):
    def test_delimiter_ids_match_the_base_models_tokenizer(self):
        # From Qwen/Qwen3.5-4B-Base's tokenizer.json added_tokens, confirmed by
        # its tokenizer_config.json added_tokens_decoder. These are the ids
        # AutoTokenizer.from_pretrained resolves.
        self.assertEqual(dict(zip(kev_prompt.SPECIAL, kev_prompt.SPECIAL_IDS)), {
            "<|fim_prefix|>": 248060,
            "<|fim_middle|>": 248061,
            "<|box_start|>": 248049,
            "<|box_end|>": 248050,
            "<|fim_suffix|>": 248062,
        })

    def test_ids_are_not_the_stale_ones_from_added_tokens_json(self):
        # kev-4b/added_tokens.json maps these to 151xxx, the Qwen2.5 range.
        # Qwen3.5-4B-Base has no token between 151600 and 151700, so using them
        # would read unrelated embedding rows.
        stale = {151659, 151660, 151648, 151649, 151661}
        self.assertFalse(stale & set(kev_prompt.SPECIAL_IDS),
                         "delimiter ids must not come from the stale added_tokens.json")

    def test_ids_sit_above_the_base_vocabulary(self):
        # They are added tokens of the base model (vocab size 248044), so a real
        # embedding table indices them directly.
        self.assertTrue(all(identifier >= 248044 for identifier in kev_prompt.SPECIAL_IDS))

    def test_training_budgets_are_the_ones_encode_carries(self):
        self.assertEqual(kev_prompt.MAX_STATE, 384)
        self.assertEqual(kev_prompt.MAX_BRANCH, 1024)
        self.assertEqual(kev_prompt.MAX_PACKED, 2048)


class SanitiseTest(unittest.TestCase):
    def test_a_delimiter_in_caller_text_is_rewritten(self):
        self.assertEqual(kev_prompt.sanitise("a <|box_end|> b"), "a <¦box_end¦> b")

    def test_several_delimiters_are_all_rewritten(self):
        self.assertEqual(kev_prompt.sanitise("<|fim_prefix|><|fim_suffix|>"),
                         "<¦fim_prefix¦><¦fim_suffix¦>")

    def test_ordinary_angle_brackets_are_untouched(self):
        self.assertEqual(kev_prompt.sanitise("a < b > c"), "a < b > c")

    def test_is_hybrid_detects_gated_deltanet(self):
        class Config:
            layer_types = ["linear_attention", "full_attention"]

        class Plain:
            layer_types = ["full_attention"]

        self.assertTrue(kev_prompt.is_hybrid(Config()))
        self.assertFalse(kev_prompt.is_hybrid(Plain()))

    def test_is_hybrid_tolerates_a_config_without_layer_types(self):
        self.assertFalse(kev_prompt.is_hybrid(object()))

    def test_the_pattern_is_the_one_upstream_uses(self):
        # A faithful port rather than a stronger sanitiser: this is upstream's
        # pattern verbatim.
        self.assertEqual(kev_prompt._SPECIAL_RE.pattern, r"<\|([A-Za-z0-9_]+)\|>")

    def test_all_five_structural_delimiters_are_covered(self):
        for token in kev_prompt.SPECIAL:
            self.assertNotEqual(kev_prompt.sanitise(token), token,
                                "%s must not survive sanitising" % token)

    def test_the_other_delimiter_shaped_tokens_are_left_alone(self):
        # Qwen3.5-4B-Base also carries <tool_call> (248058) and </tool_call>
        # (248059). They are not structural in this layout and upstream does not
        # rewrite them, so this records the boundary of what sanitising promises.
        tokenizer = StubTokenizer()
        for token in ("<tool_call>", "</tool_call>"):
            self.assertEqual(kev_prompt.sanitise(token), token)
            # Not being rewritten does not make them structural: one option in,
            # one readout row out.
            enc = kev_prompt.encode(tokenizer, record(state="s", questions=[
                {"instr": "i", "options": [token], "label": "x"}]))
            self.assertEqual(len(enc["opt_idx"][0]), 1)
            self.assertEqual(enc["ids"][enc["decide_idx"][0]], kev_prompt.SPECIAL_IDS[4])


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tok = StubTokenizer()

    def test_state_opens_with_fim_prefix_at_position_zero(self):
        enc = kev_prompt.encode(self.tok, record(state="ab"))
        self.assertEqual(enc["ids"][0], kev_prompt.SPECIAL_IDS[0])
        self.assertEqual(enc["pos"][0], 0)
        self.assertEqual(enc["seg"][0], 0)
        self.assertEqual(enc["opt"][0], kev_prompt.OPT_NONE)

    def test_one_question_one_option_has_the_documented_order(self):
        enc = kev_prompt.encode(self.tok, record(state="ab", questions=[
            {"instr": "i", "options": ["o"], "label": "l"}]))
        # <|fim_prefix|> a b | <|fim_middle|> i | <|box_start|> o <|box_end|> | <|fim_suffix|>
        self.assertEqual(enc["ids"], [kev_prompt.SPECIAL_IDS[0], 1000 + ord("a") % 1000, 1000 + ord("b") % 1000,
                                      kev_prompt.SPECIAL_IDS[1], 1000 + ord("i") % 1000,
                                      kev_prompt.SPECIAL_IDS[2], 1000 + ord("o") % 1000, kev_prompt.SPECIAL_IDS[3],
                                      kev_prompt.SPECIAL_IDS[4]])
        self.assertEqual(enc["seg"], [0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_the_head_reads_box_end_for_each_option(self):
        enc = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["aa", "b"], "label": "x"}]))
        for question, ends in enumerate(enc["opt_idx"]):
            for end in ends:
                self.assertEqual(enc["ids"][end], kev_prompt.SPECIAL_IDS[3], "box_end is the readout row")

    def test_the_head_reads_fim_suffix_to_decide(self):
        enc = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["a"], "label": "x"},
            {"instr": "j", "options": ["b"], "label": "y"}]))
        self.assertEqual(len(enc["decide_idx"]), 2)
        for index in enc["decide_idx"]:
            self.assertEqual(enc["ids"][index], kev_prompt.SPECIAL_IDS[4])

    def test_option_indices_map_tokens_to_their_option(self):
        enc = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["aa", "b"], "label": "x"}]))
        # state with delimiter: 2, instr: 2, option0 (box_start+2+box_end): 4,
        # option1: 3, decide: 1.
        self.assertEqual(enc["opt"], [kev_prompt.OPT_NONE,
                                      kev_prompt.OPT_NONE,
                                      kev_prompt.OPT_NONE, kev_prompt.OPT_NONE,
                                      0, 0, 0, 0,
                                      1, 1, 1,
                                      kev_prompt.OPT_DECIDE])
        self.assertEqual(len(enc["opt"]), len(enc["ids"]))

    def test_branch_positions_restart_after_the_state(self):
        enc = kev_prompt.encode(self.tok, record(state="abc", questions=[
            {"instr": "i", "options": ["a"], "label": "x"},
            {"instr": "j", "options": ["b"], "label": "y"}]))
        state_length = 4  # <|fim_prefix|> + 3 characters
        first = enc["pos"][state_length]
        self.assertEqual(first, state_length)
        # The second question's instruction starts where the first one did.
        second_start = enc["seg"].index(2)
        self.assertEqual(enc["pos"][second_start], state_length)

    def test_labels_are_passed_through_in_order(self):
        enc = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["a"], "label": "first"},
            {"instr": "j", "options": ["b"], "label": "second"}]))
        self.assertEqual(enc["labels"], ["first", "second"])

    def test_segment_ids_separate_state_and_questions(self):
        enc = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["a"], "label": "x"},
            {"instr": "j", "options": ["b"], "label": "y"}]))
        self.assertEqual(sorted(set(enc["seg"])), [0, 1, 2])
        self.assertEqual(len(enc["seg"]), len(enc["ids"]))
        self.assertEqual(len(enc["pos"]), len(enc["ids"]))
        self.assertEqual(len(enc["opt"]), len(enc["ids"]))

    def test_option_isolation_shares_positions_across_options(self):
        one = record(state="s", questions=[
            {"instr": "i", "options": ["aaaa", "b", "cc"], "label": "x"}])
        enc = kev_prompt.encode(self.tok, one, option_isolation=True)
        positions = [pos for pos, opt in zip(enc["pos"], enc["opt"]) if opt == 0]
        other = [pos for pos, opt in zip(enc["pos"], enc["opt"]) if opt == 1]
        self.assertEqual(positions[0], other[0], "isolated options share a start position")

        # <decide> sits one position after the longest span: state length plus the
        # instruction block plus the longest option's token count.
        state_len = len(enc["ids"][:enc["seg"].index(1)])
        instr_len = 2  # <|fim_middle|> plus one character
        longest = len([kev_prompt.SPECIAL_IDS[2], 1000 + ord("a") % 1000, 1000 + ord("a") % 1000,
                       1000 + ord("a") % 1000, 1000 + ord("a") % 1000, kev_prompt.SPECIAL_IDS[3]])
        self.assertEqual(enc["pos"][enc["decide_idx"][0]],
                         state_len + instr_len + longest)

    def test_option_isolation_is_recorded(self):
        enc = kev_prompt.encode(self.tok, record(), option_isolation=True)
        self.assertTrue(enc["option_isolation"])

    def test_option_isolation_with_no_options_is_reported_not_crashed(self):
        # Upstream's max() over an empty sequence raises ValueError here.
        with self.assertRaises(kev_prompt.ContextOverflow):
            kev_prompt.encode(self.tok, record(state="s", questions=[
                {"instr": "i", "options": [], "label": "x"}]), option_isolation=True)


class ForgeryTest(unittest.TestCase):
    """Caller text must not be able to create structure."""

    def setUp(self):
        self.tok = StubTokenizer()

    def test_a_forged_box_end_in_an_option_does_not_create_a_readout_row(self):
        honest = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["safe"], "label": "x"}]))
        forged = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["<|box_end|>"], "label": "x"}]))
        # One option in, one readout row out, even though the text contains a delimiter.
        self.assertEqual(len(forged["opt_idx"][0]), 1)
        self.assertEqual(len(forged["decide_idx"]), len(honest["decide_idx"]))
        self.assertNotIn(kev_prompt.SPECIAL_IDS[3], [forged["ids"][i] for i in forged["opt_idx"][0][:-1]])

    def test_a_forged_delimiter_in_the_state_stays_inside_the_state(self):
        enc = kev_prompt.encode(self.tok, record(state="<|fim_middle|>", questions=[
            {"instr": "i", "options": ["a"], "label": "x"}]))
        # Only the real question delimiter carries segment 1's opening id.
        self.assertEqual(enc["seg"][1], 0, "forged delimiter is still state")
        self.assertEqual(enc["ids"][1], 1000 + ord("<") % 1000)

    def test_a_forged_delimiter_in_an_option_cannot_be_read_as_box_start(self):
        enc = kev_prompt.encode(self.tok, record(state="s", questions=[
            {"instr": "i", "options": ["<|fim_suffix|>"], "label": "x"}]))
        self.assertNotEqual(enc["ids"][enc["decide_idx"][0] - 1], kev_prompt.SPECIAL_IDS[4])


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.tok = StubTokenizer()

    def test_a_long_state_is_truncated_and_flagged(self):
        enc = kev_prompt.encode(self.tok, record(state="x" * 1000))
        self.assertTrue(enc["state_truncated"])
        self.assertEqual(len(enc["ids"][:kev_prompt.MAX_STATE]), kev_prompt.MAX_STATE)

    def test_a_short_state_is_not_flagged(self):
        enc = kev_prompt.encode(self.tok, record(state="short"))
        self.assertFalse(enc["state_truncated"])

    def test_strict_mode_raises_on_an_oversized_state(self):
        with self.assertRaises(kev_prompt.ContextOverflow):
            kev_prompt.encode(self.tok, record(state="x" * 1000), strict=True)

    def test_strict_mode_raises_on_an_oversized_branch(self):
        with self.assertRaises(kev_prompt.ContextOverflow):
            kev_prompt.encode(self.tok, record(state="s", questions=[
                {"instr": "i", "options": ["y" * 2000], "label": "x"}]), strict=True)

    def test_fits_reports_false_rather_than_raising(self):
        self.assertFalse(kev_prompt.fits(record(state="x" * 5000), self.tok))

    def test_fits_reports_true_for_a_small_record(self):
        self.assertTrue(kev_prompt.fits(record(state="s"), self.tok))

    def test_fits_requires_every_tokenizer_to_agree(self):
        other = StubTokenizer()

        class Never:
            def convert_tokens_to_ids(self, token):
                return 0

            def __call__(self, text, add_special_tokens=False):
                return Batch(input_ids=[0] * 5000)

        self.assertTrue(kev_prompt.fits(record(state="s"), self.tok, other))
        self.assertFalse(kev_prompt.fits(record(state="s"), self.tok, Never()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
