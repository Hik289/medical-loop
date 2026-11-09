"""Pure task metrics and clustered uncertainty estimates; no API calls."""
from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Sequence, Optional
LABELS = ['A', 'B', 'C']
LABEL_IDX = {l: i for (i, l) in enumerate(LABELS)}
UNRESOLVED = 'Unresolved'

def record_accuracy(pred: Sequence[str], gold: Sequence[str]) -> dict:
    """Record accuracy, macro-F1, and per-class recall for labels A, B, and C."""
    assert len(pred) == len(gold) and len(gold) > 0
    p = np.array([LABEL_IDX.get(x, -1) for x in pred])
    g = np.array([LABEL_IDX[x] for x in gold])
    acc = float((p == g).mean())
    recall = {}
    f1s = []
    for (i, lab) in enumerate(LABELS):
        support = int((g == i).sum())
        tp = int(((p == i) & (g == i)).sum())
        pred_pos = int((p == i).sum())
        rec = tp / support if support > 0 else float('nan')
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 and (not np.isnan(rec)) else 0.0
        recall[lab] = rec
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {'overall_acc': acc, 'macro_f1': macro_f1, 'per_class_recall': {k: round(v, 4) if not np.isnan(v) else None for (k, v) in recall.items()}, 'aki_recall': recall['C'], 'n': len(gold), 'support': {lab: int((g == i).sum()) for (i, lab) in enumerate(LABELS)}}

def rescue_harm(base_success: Sequence[bool], hc_success: Sequence[bool], gold: Optional[Sequence[str]]=None, stratify_class: Optional[str]=None) -> dict:
    """Paired rescue, harm, and net rescue, optionally restricted to a supplied class."""
    b = np.array(base_success, dtype=bool)
    h = np.array(hc_success, dtype=bool)
    assert len(b) == len(h)
    if stratify_class is not None:
        assert gold is not None
        mask = np.array([x == stratify_class for x in gold])
        (b, h) = (b[mask], h[mask])
    n = len(b)
    if n == 0:
        return {'rescue': float('nan'), 'harm': float('nan'), 'net_rescue': float('nan'), 'n_pairs': 0}
    rescue = float((~b & h).sum() / n)
    harm = float((b & ~h).sum() / n)
    return {'rescue': rescue, 'harm': harm, 'net_rescue': rescue - harm, 'n_pairs': n, 'n_rescue': int((~b & h).sum()), 'n_harm': int((b & ~h).sum())}

def score_review(pred_claim: str, ref: dict) -> dict:
    """Score a supported claim or an explicit unresolved judgment against the evidence rubric."""
    acceptable = set(ref.get('acceptable', []))
    allow_unresolved = bool(ref.get('allow_unresolved', False))
    if allow_unresolved:
        if pred_claim == UNRESOLVED:
            return {'score': 1.0, 'unsupported': 0}
        else:
            return {'score': 0.0, 'unsupported': 1}
    elif pred_claim == UNRESOLVED:
        return {'score': 0.0, 'unsupported': 0}
    elif pred_claim in acceptable:
        return {'score': 1.0, 'unsupported': 0}
    else:
        return {'score': 0.0, 'unsupported': 0}

def evidence_review_aggregate(preds: Sequence[str], refs: Sequence[dict]) -> dict:
    """Average evidence-review quality and unsupported-inference rate."""
    assert len(preds) == len(refs) and len(refs) > 0
    per = [score_review(p, r) for (p, r) in zip(preds, refs)]
    scores = [x['score'] for x in per]
    unsup = [x['unsupported'] for x in per]
    return {'evidence_review_score': float(np.mean(scores)), 'unsupported_inference_rate': float(np.mean(unsup)), 'n': len(preds), 'n_unsupported': int(sum(unsup))}

def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence, n_boot: int=10000, alpha: float=0.05, seed: int=42) -> dict:
    """Bootstrap a mean by resampling complete clusters with replacement."""
    vals = np.asarray(values, dtype=float)
    clus = np.asarray(clusters)
    uniq = np.unique(clus)
    rng = np.random.default_rng(seed)
    idx_by_cluster = {c: np.where(clus == c)[0] for c in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        picked = np.concatenate([idx_by_cluster[c] for c in sampled])
        boots[b] = vals[picked].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {'point': float(vals.mean()), 'ci_lo': lo, 'ci_hi': hi, 'n_boot': n_boot, 'n_clusters': len(uniq)}

def net_rescue_bootstrap_ci(base_success, hc_success, clusters, n_boot: int=10000, seed: int=42) -> dict:
    """Cluster bootstrap interval for paired correction minus harm."""
    b = np.array(base_success, dtype=bool)
    h = np.array(hc_success, dtype=bool)
    contrib = (~b & h).astype(float) - (b & ~h).astype(float)
    return cluster_bootstrap_ci(contrib, clusters, n_boot=n_boot, seed=seed)
