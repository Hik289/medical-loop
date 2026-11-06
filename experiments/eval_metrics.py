#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py — HumanChoice 评估指标 (pipeline 唯一评分源头)

对应 HumanLoop.md §14 + hypothesis_tree H0.anchor_2 + theorist 要求(b)。

核心指标:
  - record_accuracy: 三分类 (A/B/C) overall acc + macro-F1 + per-class recall (AKI 单列)
  - rescue / harm / net_rescue: 配对任务 (Base vs HumanChoice), 可按类分层
  - evidence_review_score: Evidence-Grounded Review 证据支持打分 [0,1]
  - unsupported_inference: 要求(b) — 证据不足却过度断言 计 Unsupported; 正确标 Unresolved 计高分
  - bootstrap_ci: cluster bootstrap 95% CI (按患者重采样, test 仅 37 AKI → CI 宽)

本模块是纯函数, 不调任何 API, 不碰网络。anchor_2 通过构造 toy GT 断言这些函数输出 == 手算值。
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Sequence, Optional

LABELS = ["A", "B", "C"]          # A=晚期CKD/ESKD, B=CKD, C=AKI(少数类)
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}
UNRESOLVED = "Unresolved"


# ---------------------------------------------------------------------------
# 1. 三分类 Record Accuracy + macro-F1 + per-class recall (评估口径硬约束)
# ---------------------------------------------------------------------------
def record_accuracy(pred: Sequence[str], gold: Sequence[str]) -> dict:
    """三分类评估。禁止只报 overall acc → 并列 macro-F1 + per-class recall(AKI=C 单列)。"""
    assert len(pred) == len(gold) and len(gold) > 0
    p = np.array([LABEL_IDX.get(x, -1) for x in pred])
    g = np.array([LABEL_IDX[x] for x in gold])
    acc = float((p == g).mean())
    # per-class recall = TP / support(该类真实样本)
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
        "aki_recall": recall["C"],           # 少数类单列
        "n": len(gold),
        "support": {lab: int((g == i).sum()) for i, lab in enumerate(LABELS)},
    }


# ---------------------------------------------------------------------------
# 2. Rescue / Harm / Net Rescue (配对任务, 可按类分层)
# ---------------------------------------------------------------------------
def rescue_harm(base_success: Sequence[bool], hc_success: Sequence[bool],
                gold: Optional[Sequence[str]] = None,
                stratify_class: Optional[str] = None) -> dict:
    """
    Rescue = #(Base 失败 & HC 成功) / N_pairs
    Harm   = #(Base 成功 & HC 失败) / N_pairs
    Net Rescue = Rescue - Harm
    分母 = 全部配对任务 (HumanLoop §14)。stratify_class 指定则只在该真类子集上算。
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
# 3. Evidence Review Score + Unsupported Inference (要求(b) 硬约束)
# ---------------------------------------------------------------------------
def score_review(pred_claim: str, ref: dict) -> dict:
    """
    单条 Evidence-Grounded Review 打分。要求(b):
      - 证据不足(ref['allow_unresolved']=True)时:
          pred == 'Unresolved'        → 正确 → score=1.0, unsupported=0
          pred == 具体断言(非 Unresolved) → 过度断言 → score=0.0, unsupported=1  ← Unsupported Inference
      - 证据充足(allow_unresolved=False)时:
          pred ∈ acceptable           → score=1.0, unsupported=0
          pred == 'Unresolved'        → 本可判却弃判 → score=0.0, unsupported=0 (保守错误, 非过度断言)
          pred ∉ acceptable           → score=0.0, unsupported=0 (普通错误)
    """
    acceptable = set(ref.get("acceptable", []))
    allow_unresolved = bool(ref.get("allow_unresolved", False))
    if allow_unresolved:
        if pred_claim == UNRESOLVED:
            return {"score": 1.0, "unsupported": 0}
        else:
            return {"score": 0.0, "unsupported": 1}     # 证据不足却断言 → Unsupported Inference
    else:
        if pred_claim == UNRESOLVED:
            return {"score": 0.0, "unsupported": 0}      # 本可判却弃判, 不算过度断言
        elif pred_claim in acceptable:
            return {"score": 1.0, "unsupported": 0}
        else:
            return {"score": 0.0, "unsupported": 0}      # 普通判错, 不算过度断言


def evidence_review_aggregate(preds: Sequence[str], refs: Sequence[dict]) -> dict:
    """批量 Evidence Review: 平均 score + Unsupported Inference 率。"""
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
# 4. Cluster bootstrap 95% CI (按患者重采样)
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence,
                         n_boot: int = 10000, alpha: float = 0.05,
                         seed: int = 42) -> dict:
    """
    按 cluster(患者) 重采样的 bootstrap CI。同一参与者/患者多次回答有相关性 → cluster 级重采样。
    values: 每样本的值 (如 net_rescue 的逐样本贡献, 或 0/1 success)。
    clusters: 每样本所属 cluster id (患者)。
    """
    vals = np.asarray(values, dtype=float)
    clus = np.asarray(clusters)
    uniq = np.unique(clus)
    rng = np.random.default_rng(seed)
    # 预建 cluster → indices
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
    """Net Rescue 的 cluster bootstrap CI。逐样本贡献 = 1[rescue] - 1[harm]。
    G1 判据: CI 下界 > 0。"""
    b = np.array(base_success, dtype=bool)
    h = np.array(hc_success, dtype=bool)
    contrib = (((~b) & h).astype(float)) - ((b & (~h)).astype(float))
    return cluster_bootstrap_ci(contrib, clusters, n_boot=n_boot, seed=seed)
