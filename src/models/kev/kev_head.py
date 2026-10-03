#!/usr/bin/env python3
"""Kev's pointer head and its decision readout.

Ported from ``PointerHead`` and ``DecisionModel.probs`` in upstream's
``kev/model.py``. The head is a bilinear pointer: logits are
``scale * (k(h_opts) @ q(h_decide))`` with ``scale = 1/sqrt(dp)``, divided by the
checkpoint's fitted temperature at inference only, then softmaxed over the
question's candidates.

This is deliberately numpy, not a tensor library: the head is 1,311,232
parameters against a 4.2B-parameter backbone, so it is not worth a kernel, and
keeping it in fp32 follows the same reasoning as #19's readout. It runs on the
host, off the GPU, on the few vectors the encoder produced.
"""

from __future__ import annotations

import math

import numpy as np


def softmax(logits):
    """Numerically stable softmax over the last axis."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


class PointerHead:
    """The trained readout: two projections and a scaled dot product.

    ``temperature`` is fitted per checkpoint on development rows and is applied
    in eval mode only. It divides the logits, so it changes the probabilities and
    not the argmax.
    """

    def __init__(self, q_weight, q_bias, k_weight, k_bias, temperature=1.0):
        self.q_weight = np.asarray(q_weight, dtype=np.float32)
        self.q_bias = np.asarray(q_bias, dtype=np.float32)
        self.k_weight = np.asarray(k_weight, dtype=np.float32)
        self.k_bias = np.asarray(k_bias, dtype=np.float32)
        if self.q_weight.shape != self.k_weight.shape:
            raise ValueError("q and k weights must have the same shape, got %s and %s"
                             % (self.q_weight.shape, self.k_weight.shape))
        self.head_dim = self.q_weight.shape[0]
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.temperature = float(temperature)

    @property
    def parameters(self):
        return (self.q_weight.size + self.q_bias.size
                + self.k_weight.size + self.k_bias.size)

    def _project(self, weight, bias, hidden):
        return np.asarray(hidden, dtype=np.float32) @ weight.T + bias

    def logits(self, h_decide, h_opts, apply_temperature=True):
        """Logits for one question: ``h_decide`` is [d], ``h_opts`` is [K, d]."""
        options = np.atleast_2d(np.asarray(h_opts, dtype=np.float32))
        query = self._project(self.q_weight, self.q_bias, h_decide)
        keys = self._project(self.k_weight, self.k_bias, options)
        scores = (keys @ query) * self.scale
        if apply_temperature and self.temperature != 1.0:
            scores = scores / self.temperature
        return scores

    def logits_many(self, h_decide, h_opts, owner, apply_temperature=True):
        """Logits for several questions at once.

        ``h_decide`` is [Q, d], ``h_opts`` is [sum K, d] and ``owner`` maps each
        option row to its question. This is upstream's ``many()``: one pass
        instead of one per question.
        """
        decides = np.atleast_2d(np.asarray(h_decide, dtype=np.float32))
        options = np.atleast_2d(np.asarray(h_opts, dtype=np.float32))
        owners = np.asarray(owner, dtype=np.int64)
        if options.shape[0] != owners.shape[0]:
            raise ValueError("owner must have one entry per option row, got %d and %d"
                             % (owners.shape[0], options.shape[0]))
        queries = self._project(self.q_weight, self.q_bias, decides)
        keys = self._project(self.k_weight, self.k_bias, options)
        scores = np.sum(keys * queries[owners], axis=-1) * self.scale
        if apply_temperature and self.temperature != 1.0:
            scores = scores / self.temperature
        return scores

    def probs(self, h_decide, h_opts, apply_temperature=True):
        """Probabilities over the candidates of one question."""
        return softmax(self.logits(h_decide, h_opts, apply_temperature))

    def probs_by_question(self, h_decide, h_opts, owner, apply_temperature=True):
        """Probabilities per question, for the batched form.

        Returns a list of arrays, one per question, in question order. Softmax is
        taken within each question: upstream's probabilities are relative to the
        candidate set in the request, not to the whole batch.
        """
        scores = self.logits_many(h_decide, h_opts, owner, apply_temperature)
        owners = np.asarray(owner, dtype=np.int64)
        result = []
        for question in range(int(owners.max()) + 1 if owners.size else 0):
            result.append(softmax(scores[owners == question]))
        return result


def load_head(path):
    """Load a ``head.pt`` checkpoint into a :class:`PointerHead`."""
    from checkpoint import load

    payload = load(path)
    weights = payload["head"]
    return PointerHead(
        q_weight=weights["q.weight"].numpy(),
        q_bias=weights["q.bias"].numpy(),
        k_weight=weights["k.weight"].numpy(),
        k_bias=weights["k.bias"].numpy(),
        temperature=payload.get("temperature", 1.0),
    )
