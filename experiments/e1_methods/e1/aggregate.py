"""Honest aggregation of per-task records.

* Metrics are means over tasks with status "measured" only; error/timeout/missing tasks are
  counted and listed, never imputed. A cell is "complete" only if every expected task is measured.
* Paired comparisons (Rescue/Harm/Net Rescue, delta) use the intersection of tasks measured
  for both methods, with a task-resampling bootstrap 95% CI.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

from .config import METHODS
from .runner import task_universe
from .store import read_json, write_json

PRIMARY = {"clarifybench": ["param_match_rate", "tool_match_rate", "success", "exact_match",
                            "execution_success_rate", "coverage_rate"],
           "hilsql": ["solve", "ask_precision", "ask_recall", "ask_f1"]}
BINARY = {"clarifybench": "success", "hilsql": "solve"}
GRADED = {"clarifybench": "param_match_rate", "hilsql": "solve"}
# non-default respondents get their own cells "<proto>/<method>@<tag>" (official_sim cells keep their names)
RESPONDENT_TAGS = {"clarifybench": ["", "gold_intention"], "hilsql": [""]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def boot_ci(xs: List[float], n: int = 2000, seed: int = 0):
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    ms = sorted(_mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


def load_cell(out_dir, bench, proto, method) -> Dict[str, Dict[str, Any]]:
    recs = {}
    for f in glob.glob(os.path.join(out_dir, bench, proto, method, "tasks", "*.json")):
        r = read_json(f)
        if r and r.get("task_id") and not r.get("forced_branch"):
            recs[r["task_id"]] = r
    return recs


def summarize_cell(bench, recs, expected: List[str]) -> Dict[str, Any]:
    st = {s: [] for s in ("measured", "error", "timeout")}
    for t in expected:
        r = recs.get(t)
        if r is not None and r.get("status") in st:
            st[r["status"]].append(t)
    missing = [t for t in expected if t not in recs]
    meas = [recs[t] for t in st["measured"]]
    metrics = {}
    for k in PRIMARY[bench]:
        xs = [r["metrics"][k] for r in meas if r.get("metrics", {}).get(k) is not None]
        metrics[k] = {"mean": _mean(xs), "n": len(xs), "ci95": boot_ci(xs)}

    def api(role=None, key="total_tokens"):
        xs = []
        for r in meas:
            a = r.get("api") or {}
            xs.append((a.get("total") or {}).get(key, 0) if role is None else
                      ((a.get("by_role") or {}).get(role) or {}).get(key, 0))
        return _mean(xs)
    usd = [r["api"]["usd"] for r in meas if (r.get("api") or {}).get("usd") is not None]
    return {
        "n_expected": len(expected), "n_measured": len(st["measured"]), "n_error": len(st["error"]),
        "n_timeout": len(st["timeout"]), "n_missing": len(missing),
        "complete": len(st["measured"]) == len(expected) and len(expected) > 0,
        "metrics": metrics,
        "questions_per_task": _mean([float(r.get("controller", {}).get("counters", {}).get("questions", 0)) for r in meas]),
        "user_questions_per_task_official_count": _mean([float(r.get("n_user_questions") or 0) for r in meas]),
        "api_calls_per_task": api(key="live_calls"),
        "api_tokens_per_task": api(),
        "api_tokens_per_task_by_role": {ro: api(ro) for ro in ("agent", "method", "user_sim", "judge")},
        "api_usd_per_task": _mean(usd) if usd and len(usd) == len(meas) else None,
        "human_seconds": None,
        **_gold_stats(meas),
        "error_tasks": st["error"][:50], "timeout_tasks": st["timeout"][:50], "missing_tasks": missing[:50],
    }


def _gold_stats(meas: List[Dict[str, Any]]) -> Dict[str, Any]:
    g = [r["gold_intention"] for r in meas if isinstance(r.get("gold_intention"), dict)]
    if not g:
        return {}
    return {"gold_intention": {
        "n_tasks": len(g),
        "n_tasks_aligned": sum(1 for x in g if x.get("aligned")),
        "n_tasks_with_scripted_clarification": sum(1 for x in g if x.get("n_clarifications", 0) > 0),
        "n_replies": sum(x.get("n_replies", 0) for x in g),
        "n_replies_with_gold": sum(x.get("n_replies_with_gold", 0) for x in g),
        "n_tasks_gold_revealed": sum(1 for x in g if x.get("n_replies_with_gold", 0) > 0)}}


def paired(bench, ref: Dict[str, Any], cur: Dict[str, Any], seed: int = 0) -> Optional[Dict[str, Any]]:
    common = sorted(t for t in ref if t in cur and ref[t].get("status") == "measured" and cur[t].get("status") == "measured")
    if not common:
        return None
    b, g = BINARY[bench], GRADED[bench]
    rows = []
    for t in common:
        rb, cb = ref[t]["metrics"].get(b), cur[t]["metrics"].get(b)
        rg, cg = ref[t]["metrics"].get(g), cur[t]["metrics"].get(g)
        if None in (rb, cb, rg, cg):
            continue
        rows.append((1.0 if (rb < 1 and cb >= 1) else 0.0, 1.0 if (rb >= 1 and cb < 1) else 0.0, cg - rg,
                     1.0 if cg > rg else 0.0, 1.0 if cg < rg else 0.0))
    if not rows:
        return None
    n = len(rows)
    rescue, harm = _mean([r[0] for r in rows]), _mean([r[1] for r in rows])
    g_rescue, g_harm = _mean([r[3] for r in rows]), _mean([r[4] for r in rows])
    rng = random.Random(seed)
    nets, dels, gnets = [], [], []
    for _ in range(2000):
        smp = [rows[rng.randrange(n)] for _ in range(n)]
        nets.append(_mean([s[0] - s[1] for s in smp]))
        dels.append(_mean([s[2] for s in smp]))
        gnets.append(_mean([s[3] - s[4] for s in smp]))
    nets.sort(), dels.sort(), gnets.sort()
    return {"n_common": n, "binary_metric": b, "rescue": rescue, "harm": harm, "net_rescue": rescue - harm,
            "net_rescue_ci95": [nets[50], nets[1949]], "graded_metric": g, "delta_mean": _mean([r[2] for r in rows]),
            "delta_ci95": [dels[50], dels[1949]],
            # graded Rescue/Harm: fraction of tasks whose graded metric strictly rises / falls vs the reference
            "graded_rescue": g_rescue, "graded_harm": g_harm, "graded_net_rescue": g_rescue - g_harm,
            "graded_net_rescue_ci95": [gnets[50], gnets[1949]]}


def aggregate(cfg: Dict[str, Any], split: str = "test") -> Dict[str, Any]:
    out_dir = cfg["run"]["out_dir"]
    res: Dict[str, Any] = {"split": split, "honesty": "Means over status=measured tasks only; no imputation, "
                           "no modelling. Incomplete cells are flagged complete=false.", "benchmarks": {}}
    for bench in ("clarifybench", "hilsql"):
        try:
            uni = task_universe(cfg, bench)
        except Exception as e:  # noqa: BLE001 - benchmark root not present
            res["benchmarks"][bench] = {"unavailable": str(e)}
            continue
        expected = sorted(t for t, v in uni.items() if v["available"] and (split == "all" or v["split"] == split))
        unavailable = sorted(t for t, v in uni.items() if not v["available"])
        unavailable_reasons = {t: uni[t].get("reason") for t in unavailable}
        ref = load_cell(out_dir, bench, "native", "autonomous")
        b = {"n_expected_tasks": len(expected), "unavailable_tasks": unavailable,
             "unavailable_reasons": unavailable_reasons, "cells": {}}
        for proto in ("native", "choice"):
            for m in METHODS:
                for tag in RESPONDENT_TAGS[bench]:
                    name = m + (f"@{tag}" if tag else "")
                    recs = load_cell(out_dir, bench, proto, name)
                    if not recs:
                        continue
                    cell = summarize_cell(bench, recs, expected)
                    cell["respondent"] = tag or ("official_sim" if bench == "clarifybench" else "official")
                    if m != "autonomous":
                        cell["paired_vs_autonomous"] = paired(bench, {t: ref[t] for t in expected if t in ref},
                                                              {t: recs[t] for t in expected if t in recs})
                    b["cells"][f"{proto}/{name}"] = cell
        res["benchmarks"][bench] = b
    write_json(os.path.join(out_dir, f"E1_measured_summary_{split}.json"), res)
    with open(os.path.join(out_dir, f"E1_measured_summary_{split}.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(res))
    return res


def _fmt(x, pct=True):
    if x is None:
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x:.1f}"


def to_markdown(res: Dict[str, Any]) -> str:
    lines = [f"# E1 measured results (split={res['split']})", "", res["honesty"], ""]
    for bench, b in res["benchmarks"].items():
        lines.append(f"## {bench}")
        if "unavailable" in b:
            lines += [f"unavailable: {b['unavailable']}", ""]
            continue
        keys = PRIMARY[bench]
        lines.append(f"expected tasks: {b['n_expected_tasks']}; unavailable (not run): {len(b['unavailable_tasks'])}")
        lines.append("")
        lines.append("| protocol/method | measured/expected | err | t/o | " + " | ".join(keys) +
                     " | Q/task | API calls/task | tokens/task | Net Rescue vs Auto [CI] | Human s |")
        lines.append("|" + "---|" * (len(keys) + 9))
        for name, c in b["cells"].items():
            pv = c.get("paired_vs_autonomous")
            nr = f"{100 * pv['net_rescue']:+.1f} [{100 * pv['net_rescue_ci95'][0]:+.1f},{100 * pv['net_rescue_ci95'][1]:+.1f}] (n={pv['n_common']})" if pv else "—"
            flag = "" if c["complete"] else " ⚠partial"
            lines.append(f"| {name}{flag} | {c['n_measured']}/{c['n_expected']} | {c['n_error']} | {c['n_timeout']} | " +
                         " | ".join(_fmt(c["metrics"][k]["mean"]) for k in keys) +
                         f" | {_fmt(c['questions_per_task'], False)} | {_fmt(c['api_calls_per_task'], False)} | "
                         f"{_fmt(c['api_tokens_per_task'], False)} | {nr} | n/a (simulated) |")
        lines.append("")
        lines += _respondent_table(b)
    return "\n".join(lines) + "\n"


def _ci(v, ci):
    return f"{100 * v:+.1f} [{100 * ci[0]:+.1f},{100 * ci[1]:+.1f}]" if v is not None and ci else "—"


def _respondent_table(b: Dict[str, Any]) -> List[str]:
    """official_sim vs gold_intention side by side for every method that has a gold_intention cell."""
    gold = [k for k in b["cells"] if k.endswith("@gold_intention")]
    if not gold:
        return []
    out = ["### Respondent conditions (paired vs native/autonomous on commonly measured tasks)", "",
           "| protocol/method | respondent | measured/expected | param_match | Δparam_match [CI] | "
           "Net Rescue success [CI] | Graded Net Rescue param_match [CI] | gold replies / questions | "
           "tasks gold revealed |", "|" + "---|" * 9]
    for gk in gold:
        for k in (gk.split("@")[0], gk):
            c = b["cells"].get(k)
            if c is None:
                out.append(f"| {k} | official_sim | not run | — | — | — | — | — | — |")
                continue
            pv = c.get("paired_vs_autonomous") or {}
            gs = c.get("gold_intention") or {}
            out.append(
                f"| {k}{'' if c['complete'] else ' ⚠partial'} | {c.get('respondent')} | {c['n_measured']}/{c['n_expected']} | "
                f"{_fmt(c['metrics']['param_match_rate']['mean'])} | {_ci(pv.get('delta_mean'), pv.get('delta_ci95'))} | "
                f"{_ci(pv.get('net_rescue'), pv.get('net_rescue_ci95'))} | "
                f"{_ci(pv.get('graded_net_rescue'), pv.get('graded_net_rescue_ci95'))} | "
                + (f"{gs['n_replies_with_gold']}/{gs['n_replies']} | {gs['n_tasks_gold_revealed']}/{gs['n_tasks']} |"
                   if gs else "n/a | n/a |"))
    out.append("")
    return out
