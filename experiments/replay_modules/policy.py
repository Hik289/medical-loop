# -*- coding: utf-8 -*-
"""G/Q/A training (A -> Q -> G, coordinate-wise, patient-disjoint D_A/D_Q/D_G) from real
branch records, policy application, and deterministic evaluation over cached outcomes."""
from __future__ import annotations
import hashlib, math
import numpy as np
from .common import MEASURED, ERROR, SEEDS, existing
from . import features as F
from .mlp import MLP, train_bce, train_pairwise, n_params, _sig

LAMBDA_H = 0.0005   # per simulated second; pipeline.py placeholder, frozen, recorded in artifacts
MIN_SAMPLES = 4
NON_SUB = ("insufficient", "none", "skip", "no_response")


# ---------------------------------------------------------------- module container
class Modules:
    """Holds trained MLPs (or None -> documented fallback rule) + optional dev calibration."""

    def __init__(self, A=None, Q=None, G=None, calib=None, meta=None):
        self.A, self.Q, self.G, self.calib, self.meta = A, Q, G, calib or {}, meta or {}

    # fallbacks are the iteration-0 policy: Q = pool order, G = always ask, A = adopt revision
    def q_scores(self, rec):
        if self.Q is None:
            return [-float(i) for i in range(len(rec["questions"]))]
        return list(self.Q.logits(np.stack([F.q_feat(rec, q) for q in rec["questions"]])))

    def _p(self, name, logit):
        c = self.calib.get(name)
        return float(_sig(c["a"] * logit + c["b"])) if c else float(_sig(logit))

    def p_ask(self, rec, qi, scores):
        if self.G is None:
            return 1.0
        return self._p("G", float(self.G.logits(F.g_feat(rec, rec["questions"][qi], scores, qi)[None])[0]))

    def p_adopt(self, rec, q):
        if self.A is None:
            return 1.0
        return self._p("A", float(self.A.logits(F.a_feat(rec, q)[None])[0]))

    def to_dict(self):
        return {"A": self.A.to_dict() if self.A else None, "Q": self.Q.to_dict() if self.Q else None,
                "G": self.G.to_dict() if self.G else None, "calib": self.calib, "meta": self.meta,
                "schema": F.SCHEMA["name"], "schema_sha": F.SCHEMA_SHA}

    @classmethod
    def from_dict(cls, d):
        if d.get("schema_sha") != F.SCHEMA_SHA:
            raise ValueError(f"feature schema mismatch: artifact {d.get('schema_sha')} vs code {F.SCHEMA_SHA}")
        mk = lambda x: MLP.from_dict(x) if x else None  # noqa: E731
        return cls(mk(d["A"]), mk(d["Q"]), mk(d["G"]), d.get("calib"), d.get("meta"))


# ---------------------------------------------------------------- decisions
def answered(q):
    return (q.get("answer") or {}).get("status") == MEASURED and (q.get("revision") or {}).get("status") == MEASURED


def adopt(mods, rec, q, always_revise=False):
    """A with hard constraints: non-substantive answers keep a0."""
    if q["answer"]["r_t_type"] in NON_SUB or not q["revision"].get("changed"):
        return False
    return True if always_revise else mods.p_adopt(rec, q) > 0.5


def u_ask(mods, rec, q, lam=None):
    lam = LAMBDA_H if lam is None else lam
    y = q["Y_ask"] if adopt(mods, rec, q) else rec["base"]["Y"]
    return y - lam * q["est_seconds"]


def decide(mods, rec, variant="full", rng_key="v"):
    """Apply a policy to one record. Variants (E5 ablations, same trained modules):
    full | no_G (always ask) | no_Q (seeded random eligible question) | no_A (always revised)."""
    if rec.get("status") != MEASURED:
        return {"status": ERROR, "error": rec.get("error", "record not measured")}
    base = rec["base"]
    if not rec["questions"]:
        return {"status": MEASURED, "asked": False, "action": base["action"], "Y": base["Y"], "seconds": 0, "qi": None}
    scores = mods.q_scores(rec)
    if variant == "no_Q":
        h = lambda i: hashlib.sha256(f"{rng_key}|{rec['task_id']}|{i}".encode()).hexdigest()  # noqa: E731
        qi = min(range(len(rec["questions"])), key=h)
    else:
        qi = int(np.argmax(scores))
    ask = True if variant == "no_G" else mods.p_ask(rec, qi, scores) > 0.5
    if not ask:
        return {"status": MEASURED, "asked": False, "action": base["action"], "Y": base["Y"], "seconds": 0, "qi": qi}
    q = rec["questions"][qi]
    if not answered(q):
        return {"status": ERROR, "error": f"selected question {q['qid']} has no measured answer/revision", "qi": qi}
    take = adopt(mods, rec, q, always_revise=(variant == "no_A"))
    return {"status": MEASURED, "asked": True, "qi": qi, "r_t_type": q["answer"]["r_t_type"],
            "action": q["revision"]["action"] if take else base["action"],
            "Y": q["Y_ask"] if take else base["Y"], "seconds": q["est_seconds"], "adopted": take,
            "proposed_change": bool(q["revision"].get("changed"))}


# ---------------------------------------------------------------- training
def build_A(records):
    X, y, ties = [], [], 0
    for r in records:
        if r.get("status") != MEASURED or r.get("group") not in (None, "D_A"):
            continue
        for q in r["questions"]:
            if not answered(q) or q["answer"]["r_t_type"] in NON_SUB or not q["revision"].get("changed"):
                continue
            if q["Y_ask"] == r["base"]["Y"]:
                ties += 1
                continue
            X.append(F.a_feat(r, q))
            y.append(1.0 if q["Y_ask"] > r["base"]["Y"] else 0.0)
    return np.array(X), np.array(y), {"n": len(y), "ties_dropped": ties}


def build_Q(records, mods):
    P, N = [], []
    for r in records:
        if r.get("status") != MEASURED or r.get("group") not in (None, "D_Q"):
            continue
        qs = [q for q in r["questions"] if answered(q)]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                ui, uj = u_ask(mods, r, qs[i]), u_ask(mods, r, qs[j])
                if abs(ui - uj) < 1e-12:
                    continue
                hi, lo = (qs[i], qs[j]) if ui > uj else (qs[j], qs[i])
                P.append(F.q_feat(r, hi))
                N.append(F.q_feat(r, lo))
    return np.array(P), np.array(N), {"n_pairs": len(P)}


def build_G(records, mods):
    X, y, skipped = [], [], 0
    for r in records:
        if r.get("status") != MEASURED or r.get("group") not in (None, "D_G") or not r["questions"]:
            continue
        scores = mods.q_scores(r)
        qi = int(np.argmax(scores))
        q = r["questions"][qi]
        if not answered(q):  # requirement (d): label only with the q_t Q actually picks
            skipped += 1
            continue
        X.append(F.g_feat(r, q, scores, qi))
        y.append(1.0 if u_ask(mods, r, q) > r["base"]["Y"] else 0.0)
    return np.array(X), np.array(y), {"n": len(y), "skipped_unanswered_q_t": skipped}


def train_modules(records, seed, epochs=30, use_groups=True):
    """Coordinate-wise A -> Q -> G. If a module has < MIN_SAMPLES training items it keeps
    its fallback rule and the log says so (no fabricated module)."""
    recs = records if use_groups else [dict(r, group=None) for r in records]
    mods, log = Modules(meta={"seed": seed, "lambda_H": LAMBDA_H, "params_per_module": n_params()}), {}
    X, y, info = build_A(recs)
    if len(y) >= MIN_SAMPLES:
        mods.A, tl = train_bce(X, y, seed, epochs)
        info.update(tl, pos_rate=float(y.mean()))
    else:
        info["fallback"] = "adopt-revision rule (insufficient A data)"
    log["A"] = info
    Pq, Nq, info = build_Q(recs, mods)
    if len(Pq) >= MIN_SAMPLES:
        mods.Q, tl = train_pairwise(Pq, Nq, seed, epochs)
        info.update(tl)
    else:
        info["fallback"] = "pool-order rule (insufficient Q pairs)"
    log["Q"] = info
    X, y, info = build_G(recs, mods)
    if len(y) >= MIN_SAMPLES:
        mods.G, tl = train_bce(X, y, seed, epochs)
        info.update(tl, pos_rate=float(y.mean()))
    else:
        info["fallback"] = "always-ask rule (insufficient G data)"
    log["G"] = info
    mods.meta["train_log"] = log
    return mods


# ---------------------------------------------------------------- evaluation
def summarize(records, outcomes, medical=True, n_boot=2000):
    """Deterministic metrics over measured outcomes; errors are counted, never imputed."""
    EM = existing("eval_metrics")
    pairs = [(r, o) for r, o in zip(records, outcomes) if o.get("status") == MEASURED]
    n_err = len(outcomes) - len(pairs)
    if not pairs:
        return {"status": ERROR, "n_total": len(outcomes), "n_error": n_err}
    base = [bool(r["base"]["Y"] >= 1.0) for r, _ in pairs]
    hc = [bool(o["Y"] >= 1.0) for _, o in pairs]
    rh = EM.rescue_harm(base, hc)
    ci = EM.net_rescue_bootstrap_ci(base, hc, [r["cluster_id"] for r, _ in pairs], n_boot=n_boot, seed=42)
    secs = [o["seconds"] for _, o in pairs]
    out = {"status": MEASURED, "n_total": len(outcomes), "n_measured": len(pairs), "n_error": n_err,
           "final_quality": round(100 * float(np.mean([o["Y"] for _, o in pairs])), 2),
           "autonomous_quality": round(100 * float(np.mean([r["base"]["Y"] for r, _ in pairs])), 2),
           "rescue": round(100 * rh["rescue"], 2), "harm": round(100 * rh["harm"], 2),
           "net_rescue": round(100 * rh["net_rescue"], 2),
           "net_rescue_ci95": [round(100 * ci["ci_lo"], 2), round(100 * ci["ci_hi"], 2)],
           "asked_pct": round(100 * float(np.mean([o["asked"] for _, o in pairs])), 2),
           "unnecessary_questions": int(sum(1 for r, o in pairs if o["asked"] and o["action"] == r["base"]["action"])),
           "simulated_seconds_mean": round(float(np.mean(secs)), 1),
           "simulated_seconds_note": "est_seconds per question type (pipeline simulated_time table), NOT measured human time"}
    out["augmentation_gain"] = round(out["final_quality"] - out["autonomous_quality"], 2)
    if medical:
        golds = [r["gold"] for r, _ in pairs]
        acc = EM.record_accuracy([o["action"] for _, o in pairs], golds)
        out.update(record_acc=round(100 * acc["overall_acc"], 2), macro_f1=round(100 * acc["macro_f1"], 2),
                   per_class_recall={k: (None if v is None else round(100 * v, 2)) for k, v in acc["per_class_recall"].items()})
        rc = EM.rescue_harm(base, hc, gold=golds, stratify_class="C")
        out["net_rescue_C_AKI"] = None if math.isnan(rc["net_rescue"]) else round(100 * rc["net_rescue"], 2)
    return out


def seed_mean_std(rows, keys=("final_quality", "rescue", "harm", "net_rescue", "asked_pct", "simulated_seconds_mean", "record_acc", "macro_f1")):
    ok = [r for r in rows if r.get("status") == MEASURED]
    if len(ok) != len(rows) or not ok:
        return {"status": ERROR, "n_seeds_measured": len(ok), "per_seed": rows}
    agg = {"status": MEASURED, "n_seeds": len(ok), "per_seed": rows}
    for k in keys:
        if k in ok[0]:
            v = [r[k] for r in ok]
            agg[k] = {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2)}
    return agg
