# -*- coding: utf-8 -*-
"""xfeat_v1 — domain-agnostic G/Q/A features (<=64 dims, zero padded, deterministic).

Defined identically for medical, NoisyToolBench and HiL-SQL checkpoint records so frozen
modules can transfer (E7). G and Q features use only pre-ask information; A features use
the response and the proposed revision but never the outcome Y or the reference.

Why not reuse the existing pipeline features: pipeline.s6_features puts 1[a0!=a1]
(only observable after the human answered and the revision ran) into the *G* vector,
so a gate trained on it cannot be applied before asking.
"""
from __future__ import annotations
import numpy as np
from .common import sha

QTYPES = ["Scope", "Rule", "Evidence", "Temporal", "Preference"]
RTYPES = ["substantive", "insufficient", "none", "skip"]
DIM = 64


def _onehot(v, vocab):
    return [1.0 if v == x else 0.0 for x in vocab]


def _unc(rec):
    u = (rec.get("base") or {}).get("uncertainty")
    try:
        return min(max(float(u), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.5  # missing self-report -> neutral; counted in feature audit


def _pad(v):
    assert len(v) <= DIM
    return np.array(v + [0.0] * (DIM - len(v)), dtype=np.float32)


def q_feat(rec, q):
    return _pad(_onehot(q["type"], QTYPES) + [
        min(len(q.get("question") or ""), 300) / 300.0,
        q["n_options"] / 3.0,
        min(q["n_evidence"], 5) / 5.0,
        q["est_seconds"] / 120.0,
        _unc(rec),
        float(rec.get("missing_ratio") or 0.0),
        float(q.get("evidence_missing_frac") or 0.0),
    ])


def g_feat(rec, q, q_scores, qi):
    """q = the question Q actually selected (requirement (d)); q_scores = Q logits of pool."""
    s = np.sort(np.asarray(q_scores, dtype=float))[::-1]
    margin = float(s[0] - s[1]) if len(s) > 1 else 0.0
    top = 1.0 / (1.0 + np.exp(-float(np.clip(q_scores[qi], -40, 40))))
    base = list(q_feat(rec, q)[:12])
    return _pad(base + [len(rec["questions"]) / 3.0, top, float(np.tanh(margin))])


def a_feat(rec, q):
    ans, rev = q["answer"], q["revision"]
    return _pad(_onehot(ans["r_t_type"], RTYPES) + [
        1.0 if rev and rev.get("changed") else 0.0,
        float(ans.get("agreement") if ans.get("agreement") is not None else 1.0),
        _unc(rec),
        float(rec.get("missing_ratio") or 0.0),
    ] + _onehot(q["type"], QTYPES) + [q["n_options"] / 3.0])


SCHEMA = {"name": "xfeat_v1", "dim": DIM,
          "Q": ["qtype_onehot5", "q_len", "n_opts", "n_evidence", "est_seconds", "base_uncertainty",
                "missing_ratio", "evidence_missing_frac"],
          "G": ["Q_feat[:12]", "pool_size", "q_top_sigmoid", "tanh_q_margin"],
          "A": ["rtype_onehot4", "changed", "participant_agreement", "base_uncertainty", "missing_ratio",
                "qtype_onehot5", "n_opts"]}
SCHEMA_SHA = sha(SCHEMA)
