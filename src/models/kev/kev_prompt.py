#!/usr/bin/env python3
"""Kev prompt packing, ported from ``kev/model.py`` in jaredpalmer/kev.

Kev is the second Qwen3.5 decision model proposed for System1-Omni (#9), and its
prompt is built programmatically: the published checkpoint has no chat template
at all. This module is that builder, kept separate from the model engine so it
can be tested without a GPU, a checkpoint or PyTorch.

Ported from ``encode()``, ``user_tokens()`` and ``is_hybrid()`` at the revision
pinned below. The layout, position scheme and index bookkeeping are reproduced
operation for operation; nothing here is inferred.

One deliberate divergence is marked in the code: upstream's ``option_isolation``
path calls ``max()`` on the span lengths without guarding an empty option list,
which raises ``ValueError: max() arg is an empty sequence``. Here that is a
``ContextOverflow``, because an empty option list is a malformed request rather
than an internal error.

Layout for one record::

    [<|fim_prefix|>] state_tokens
      per question k in 1..Q:
        [<|fim_middle|>] instr_tokens
        per option j in 0..K-1:
          [<|box_start|>] option_tokens [<|box_end|>]
        [<|fim_suffix|>]

``<|box_end|>`` and ``<|fim_suffix|>`` are the rows the pointer head reads.
"""

from __future__ import annotations

import re

# The five delimiters are pre-existing Qwen special tokens, reused so that no
# embedding row had to be added or trained (see the comment above SPECIAL in
# kev/model.py). Their ids are taken from the base model's tokenizer, which is
# what `AutoTokenizer.from_pretrained` loads.
#
# Do NOT take these from `jaredpalmer/kev-4b/added_tokens.json`: that file is
# stale and disagrees with the tokenizer the code actually loads. It maps
# <|fim_prefix|> to 151659 and friends, which is the Qwen2.5 special-token range;
# Qwen3.5-4B-Base has no token at all between 151600 and 151700, so those ids
# would read the wrong embedding rows and produce silently wrong decisions.
#
# Verified against two independent files, which agree:
#   Qwen/Qwen3.5-4B-Base: tokenizer.json added_tokens, and
#                         tokenizer_config.json added_tokens_decoder.
# kev-4b's own tokenizer.json agrees as well, but its tokenizer_config.json
# declares no added tokens at all, so the adapter repository is not a reliable
# source for them.
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>",
           "<|fim_suffix|>")
SPECIAL_IDS = (248060, 248061, 248049, 248050, 248062)

# Training-time context budgets. Serving uses much larger ones (see the README's
# note on SERVE_MAX_STATE); these are the defaults encode() itself carries.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048

OPT_NONE, OPT_DECIDE = -1, -2

# `<|name|>` is rewritten before tokenizing, so caller text can never produce a
# delimiter and option boundaries cannot be forged.
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class ContextOverflow(ValueError):
    """Raised in strict mode when a record does not fit the training context."""


def is_hybrid(config):
    """Whether a text config has Gated DeltaNet layers (Qwen3.5 does).

    Such backbones cannot honour a block-causal mask, so they run the row form:
    state once, then one branch per question. This is why Kev's serving path
    cannot use a packed layout even though the checkpoint supports one.
    """
    return "linear_attention" in set(getattr(config, "layer_types", None) or [])


def sanitise(text):
    """Return caller text with delimiter-looking sequences made inert.

    The pattern covers the ``<|name|>`` form, which is how all five structural
    delimiters are spelled. It deliberately does not touch the two other
    delimiter-shaped tokens in the base tokenizer — ``<tool_call>`` and
    ``</tool_call>`` (ids 248058 and 248059) — because they carry no structural
    meaning in this layout, and because upstream's ``user_tokens()`` uses exactly
    this pattern. Changing it here would make the port disagree with the
    reference for no measured gain. They are recorded in the tests so the choice
    is visible rather than accidental.
    """
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


def user_tokens(tokenize, text):
    """Tokenize caller-supplied text.

    ``tokenize`` is a callable taking ``(text, add_special_tokens=False)`` and
    returning an object with ``input_ids``, matching the Hugging Face fast
    tokenizer interface.
    """
    return tokenize(sanitise(text), add_special_tokens=False).input_ids


class _Ids(list):
    """Minimal stand-in for a tokenizer's BatchEncoding."""

    def __init__(self, input_ids):
        super().__init__(input_ids)
        self.input_ids = list(input_ids)


def encode(tokenizer, record, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False,
           option_isolation=False):
    """Pack one record into the ids, segment ids, positions and readout indices.

    Returns a dict with:

    ``ids``          packed token ids
    ``seg``          0 for state, k for question k
    ``pos``          position ids, restarting per question branch
    ``opt``          per-token option index: OPT_NONE, 0..K-1, or OPT_DECIDE
    ``decide_idx``   index of ``<|fim_suffix|>`` per question
    ``opt_idx``      index of ``<|box_end|>`` per option, per question
    ``labels``       the record's labels, passed through
    ``state_truncated`` whether the state hit the budget
    """
    special_ids = [tokenizer.convert_tokens_to_ids(token) for token in SPECIAL]
    state_tokens = user_tokens(tokenizer, record["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow("state exceeds %d tokens: %d" % (max_state, len(state_tokens) + 1))

    state = [special_ids[0]] + state_tokens[: max_state - 1]
    ids = list(state)
    seg = [0] * len(state)
    pos = list(range(len(state)))
    opt = [OPT_NONE] * len(state)

    _, q_id, o_id, c_id, d_id = special_ids
    decide_idx = []
    opt_idx = []

    for k, question in enumerate(record["questions"], start=1):
        instr = [q_id] + user_tokens(tokenizer, question["instr"])
        spans = [[o_id] + user_tokens(tokenizer, option) + [c_id]
                 for option in question["options"]]
        branch = instr + [token for span in spans for token in span] + [d_id]
        if len(branch) > max_branch - len(state):
            raise ContextOverflow(
                "branch too long: %d tokens with a %d-token state (row limit %d)"
                % (len(branch), len(state), max_branch))

        base = len(ids)
        p0 = len(state)
        branch_opt = ([OPT_NONE] * len(instr)
                      + [j for j, span in enumerate(spans) for _ in span]
                      + [OPT_DECIDE])
        if option_isolation:
            # Hardening over the ported source: upstream calls max() unguarded
            # here, so a question with no options raises "max() arg is an empty
            # sequence". A caller-supplied empty option list is a request error,
            # not a crash, so it is reported as one.
            if not spans:
                raise ContextOverflow("question %d has no options" % k)
            longest = max(len(span) for span in spans)
            branch_pos = (list(range(p0, p0 + len(instr)))
                          + [p0 + len(instr) + i for span in spans for i in range(len(span))]
                          + [p0 + len(instr) + longest])
        else:
            branch_pos = list(range(p0, p0 + len(branch)))

        ends = []
        cursor = len(instr)
        for span in spans:
            cursor += len(span)
            ends.append(cursor - 1)

        ids += branch
        seg += [k] * len(branch)
        pos += branch_pos
        opt += branch_opt
        decide_idx.append(base + len(branch) - 1)
        opt_idx.append([base + end for end in ends])

    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "opt": opt,
        "option_isolation": option_isolation,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "labels": [question["label"] for question in record["questions"]],
        "state_truncated": len(state_tokens) + 1 > max_state,
    }


def fits(record, *tokenizers, max_state=MAX_STATE, max_branch=MAX_BRANCH,
         max_packed=MAX_PACKED):
    """Whether a record encodes without truncation for every given tokenizer."""
    try:
        return all(len(encode(tokenizer, record, max_state=max_state, max_branch=max_branch,
                              strict=True)["ids"]) <= max_packed
                   for tokenizer in tokenizers)
    except ContextOverflow:
        return False
