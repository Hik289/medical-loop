"""HumanChoice Expert Iteration on the ClarifyBench TRAIN split (HumanLoop §8-9).

Iteration k:
  1. freeze pi_k (k=0: init policy = first pool question, Ask, take a1; k>0: modules of iter k-1)
  2. for every train task and checkpoint cp < teacher_cps of request 0, run complete official
     branches: Wait@cp, and Ask(q_i)@cp with forced a0 / a1 for each pool question i.
     Earlier checkpoints are forced Wait; later ones follow pi_k. A prompt-hash replay cache
     makes branches of one task share an identical prefix (pool, user answer) - an
     approximation of a resettable state, recorded as such (requirement (c)).
  3. A <- L_A on D_A, then Q <- L_Q on D_Q (targets use the new A), then G <- L_G on D_G
     (targets use Q's actual q_t and its measured U^Ask; requirement (d)).
Only branches with status "measured" are used; Y is the official evaluator metric.
"""
from __future__ import annotations

import copy
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np

from . import hc
from .runner import log, run_jobs, task_universe
from .store import read_json, task_path, write_json


def branch_specs(cps: int, pool: int) -> List[Dict[str, Any]]:
    out = []
    for cp in range(cps):
        out.append({"request_idx": 0, "cp": cp, "action": "wait"})
        for qi in range(pool):
            for a in ("a0", "a1"):
                out.append({"request_idx": 0, "cp": cp, "action": "ask", "q_index": qi, "a_choice": a})
    return out


def tag(b: Dict[str, Any]) -> str:
    return f"cp{b['cp']}_wait" if b["action"] == "wait" else f"cp{b['cp']}_q{b['q_index']}_{b['a_choice']}"


def collect(cfg, k: int, protocol: str, cps: int, pool: int, limit: int, prev_modules: str | None) -> List[Dict[str, Any]]:
    c = copy.deepcopy(cfg)
    hcp = c["methods"]["humanchoice"]
    if k == 0:
        hcp["policy"] = "init"
    else:
        hcp["policy"], hcp["modules_dir"] = "trained", prev_modules
    base = os.path.join(cfg["run"]["out_dir"], "teacher", f"iter{k}")
    c["run"]["out_dir"] = base  # isolate teacher records from evaluation records
    uni = task_universe(cfg, "clarifybench")
    train = sorted(t for t, v in uni.items() if v["split"] == "train")
    if limit:
        train = train[:limit]
    jobs = []
    for t in train:
        for b in branch_specs(cps, pool):
            jobs.append({"benchmark": "clarifybench", "method": "humanchoice", "protocol": protocol, "task_id": t,
                         "split": "train", "forced": b, "replay_cache": os.path.join(base, "replay", t.replace("/", "__")),
                         "out": task_path(base, "clarifybench", protocol, "humanchoice", t, tag=tag(b))})
    run_jobs(c, jobs)
    rows = []
    for j in jobs:
        r = read_json(j["out"]) or {}
        tch = r.get("teacher") or {}
        if r.get("status") != "measured" or not tch.get("branch") or tch.get("branch_invalid"):
            continue
        rows.append({"task_id": j["task_id"], "branch": j["forced"], "Y": r["Y"], "q_features": tch.get("q_features"),
                     "a_features": tch.get("a_features"), "legal": tch.get("legal")})
    log(f"iter{k}: {len(rows)}/{len(jobs)} measured, usable branches")
    return rows


def _fit(kind: str, rows, seeds, epochs, lr, bs):
    models, logs = {}, {}
    for s in seeds:
        if kind == "Q":
            m, curve = hc.train_pairwise(np.array([r["x_pos"] for r in rows]), np.array([r["x_neg"] for r in rows]),
                                         s, epochs, lr, bs)
        else:
            m, curve = hc.train_binary(np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows]),
                                       s, epochs, lr, bs)
        models[s], logs[s] = m, {"loss_curve": [round(x, 5) for x in curve], "n_params": m.n_params()}
    return models, logs


def train_iteration(branches, out_dir: str, seeds, epochs=30, lr=1e-3, bs=64) -> Dict[str, Any]:
    grp = {b["task_id"]: hc.group_of(b["task_id"]) for b in branches}
    sel = lambda g: [b for b in branches if grp[b["task_id"]] == g]  # noqa: E731
    log_ = {"order": "A->Q->G", "seeds": seeds, "groups": {g: len({b['task_id'] for b in sel(g)}) for g in ("D_A", "D_Q", "D_G")}}
    A_rows = hc.build_A(sel("D_A"))
    log_["A"] = {"n": len(A_rows)}
    if not A_rows:
        raise RuntimeError("no A training rows (need measured a0/a1 branch pairs with different Y)")
    A, log_["A"]["seeds"] = _fit("A", A_rows, seeds, epochs, lr, bs)
    Q_rows = {s: hc.build_Q(sel("D_Q"), A[s]) for s in seeds}
    log_["Q"] = {"n": {s: len(v) for s, v in Q_rows.items()}, "seeds": {}}
    Q = {}
    for s in seeds:
        if not Q_rows[s]:
            raise RuntimeError("no Q training pairs")
        m, l = _fit("Q", Q_rows[s], [s], epochs, lr, bs)
        Q[s], log_["Q"]["seeds"][s] = m[s], l[s]
    log_["G"] = {"n": {}, "seeds": {}}
    for s in seeds:
        rows = hc.build_G(sel("D_G"), A[s], Q[s])
        log_["G"]["n"][s] = len(rows)
        if not rows:
            raise RuntimeError("no G training rows")
        m, l = _fit("G", rows, [s], epochs, lr, bs)
        log_["G"]["seeds"][s] = l[s]
        for name, mod in (("A", A[s]), ("Q", Q[s]), ("G", m[s])):
            hc.save_module(os.path.join(out_dir, f"{name}_s{s}.json"), name, mod, {"seed": s})
    log_["total_params_per_seed"] = 3 * hc.N_PARAMS
    write_json(os.path.join(out_dir, "train_log.json"), log_)
    return log_


def run_ei(cfg, iters: int, protocol: str, cps: int, pool: int, seeds, limit: int = 0, epochs: int = 30) -> str:
    resp = (cfg.get("clarifybench") or {}).get("respondent") or "official_sim"
    if resp != "official_sim":  # module/teacher dirs are not respondent-tagged; never overwrite official modules
        raise RuntimeError(f"train-hc runs only with clarifybench.respondent=official_sim (got {resp!r}); "
                           "gold_intention is an evaluation-only condition")
    prev = None
    for k in range(iters):
        branches = collect(cfg, k, protocol, cps, pool, limit, prev)
        out = os.path.join(cfg["run"]["out_dir"], "hc_modules", protocol, f"iter{k}")
        write_json(os.path.join(out, "branches_index.json"),
                   [{"task_id": b["task_id"], "branch": b["branch"], "Y": b["Y"]} for b in branches])
        train_iteration(branches, out, seeds, epochs)
        log(f"iter{k}: modules -> {out}")
        prev = out
    return prev
