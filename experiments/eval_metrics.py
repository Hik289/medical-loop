#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score record predictions, paired task results, and evidence reviews.

Report accuracy, macro-F1, and class recall together. Measure rescue and harm
on the same paired tasks. For evidence reviews, distinguish a supported
answer from a claim made without enough evidence. Estimate uncertainty by
resampling whole patients or other independent groups. No network calls.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Sequence, Optional

LABELS = ["A", "B", "C"]          # A = advanced CKD/ESKD, B = CKD, C = AKI.
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}
UNRESOLVED = "Unresolved"


# ---------------------------------------------------------------------------
# Record accuracy, macro-F1, and class recall.
# ---------------------------------------------------------------------------
def record_accuracy(pred: Sequence[str], gold: Sequence[str]) -> dict:
    """Return accuracy, macro-F1, class recall, and class counts."""
    assert len(pred) == len(gold) and len(gold) > 0
    p = np.array([LABEL_IDX.get(x, -1) for x in pred])
    g = np.array([LABEL_IDX[x] for x in gold])
    acc = float((p == g).mean())
    # Class recall is true positives divided by the class count.
    recall = {}
    f1s = []
    for i, lab in enumerate(LABELS):
        support = int((g == i).sum())
        tp = int(((p == i) & (g == i)).sum())
        pred_pos = int((p == i).sum())
        rec = tp / support if support > 0 else float("nan")
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 and not np.isnan(rec) else 0.0
        recall[lab] = rec
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {
        "overall_acc": acc,
        "macro_f1": macro_f1,
        "per_class_recall": {k: (round(v, 4) if not np.isnan(v) else None) for k, v in recall.items()},
        "aki_recall": recall["C"],           # Report the AKI class separately.
        "n": len(gold),
        "support": {lab: int((g == i).sum()) for i, lab in enumerate(LABELS)},
    }


# ---------------------------------------------------------------------------
# Rescue and harm on paired tasks, optionally restricted to a class.
# ---------------------------------------------------------------------------
def rescue_harm(base_success: Sequence[bool], hc_success: Sequence[bool],
                gold: Optional[Sequence[str]] = None,
                stratify_class: Optional[str] = None) -> dict:
    """
    Count changes relative to the baseline on the same paired tasks.

    Rescue means baseline failure followed by HumanChoice success. Harm means
    baseline success followed by HumanChoice failure. Both rates use all matched
    tasks as their denominator. Optionally restrict the pairs to a record class.
    """
    b = np.array(base_success, dtype=bool)
    h = np.array(hc_success, dtype=bool)
    assert len(b) == len(h)
    if stratify_class is not None:
        assert gold is not None
        mask = np.array([x == stratify_class for x in gold])
        b, h = b[mask], h[mask]
    n = len(b)
    if n == 0:
        return {"rescue": float("nan"), "harm": float("nan"), "net_rescue": float("nan"), "n_pairs": 0}
    rescue = float(((~b) & h).sum() / n)
    harm = float((b & (~h)).sum() / n)
    return {
        "rescue": rescue,
        "harm": harm,
        "net_rescue": rescue - harm,
        "n_pairs": n,
        "n_rescue": int(((~b) & h).sum()),
        "n_harm": int((b & (~h)).sum()),
    }


# ---------------------------------------------------------------------------
# Review scores and claims made without enough evidence.
# ---------------------------------------------------------------------------
def score_review(pred_claim: str, ref: dict) -> dict:
    """
    Score a claim against the supplied review reference.

    If the reference requires Unresolved, award it full credit and flag any
    specific claim as unsupported. Otherwise award full credit to an acceptable
    claim. Returning Unresolved when a claim is supported is incorrect, but is
    not counted as an unsupported assertion.
    """
    acceptable = set(ref.get("acceptable", []))
    allow_unresolved = bool(ref.get("allow_unresolved", False))
    if allow_unresolved:
        if pred_claim == UNRESOLVED:
            return {"score": 1.0, "unsupported": 0}
        else:
            return {"score": 0.0, "unsupported": 1}     # A specific claim without enough evidence is unsupported.
    else:
        if pred_claim == UNRESOLVED:
            return {"score": 0.0, "unsupported": 0}      # Declining a supported answer is incorrect but not an unsupported claim.
        elif pred_claim in acceptable:
            return {"score": 1.0, "unsupported": 0}
        else:
            return {"score": 0.0, "unsupported": 0}      # An incorrect supported-evidence answer is not an unsupported claim.


def evidence_review_aggregate(preds: Sequence[str], refs: Sequence[dict]) -> dict:
    """Average the review scores and the rate of unsupported claims."""
    assert len(preds) == len(refs) and len(refs) > 0
    per = [score_review(p, r) for p, r in zip(preds, refs)]
    scores = [x["score"] for x in per]
    unsup = [x["unsupported"] for x in per]
    return {
        "evidence_review_score": float(np.mean(scores)),
        "unsupported_inference_rate": float(np.mean(unsup)),
        "n": len(preds),
        "n_unsupported": int(sum(unsup)),
    }


# ---------------------------------------------------------------------------
# Confidence intervals estimated by resampling complete patient groups.
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence,
                         n_boot: int = 10000, alpha: float = 0.05,
                         seed: int = 42) -> dict:
    """
    Estimate a confidence interval by resampling complete groups.

    Each value and group ID describe the same observation. Resampling whole
    groups keeps repeated observations from the same patient together.
    """
    vals = np.asarray(values, dtype=float)
    clus = np.asarray(clusters)
    uniq = np.unique(clus)
    rng = np.random.default_rng(seed)
    # Look up all observation indices for each group.
    idx_by_cluster = {c: np.where(clus == c)[0] for c in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        picked = np.concatenate([idx_by_cluster[c] for c in sampled])
        boots[b] = vals[picked].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": float(vals.mean()), "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot,
            "n_clusters": len(uniq)}


def net_rescue_bootstrap_ci(base_success, hc_success, clusters,
                            n_boot: int = 10000, seed: int = 42) -> dict:
    """
    Estimate the net-rescue confidence interval by resampling groups.

    Each task contributes +1 for rescue, -1 for harm, and 0 otherwise.
    """
    b = np.array(base_success, dtype=bool)
    h = np.array(hc_success, dtype=bool)
    contrib = (((~b) & h).astype(float)) - ((b & (~h)).astype(float))
    return cluster_bootstrap_ci(contrib, clusters, n_boot=n_boot, seed=seed)
