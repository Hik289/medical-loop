#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cb_aggregate.py — 汇总 ClarifyBench Native 结果 (anchor_5)。
官方指标: evaluation.correctness (tool_match_rate/param_match_rate/exact_match),
          evaluation.execution.success_rate, evaluation.coverage.coverage_rate,
          evaluation.success (task success)。按子集 A/E/I 分别 + 总体。"""
import argparse, json, glob, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np

JST = timezone(timedelta(hours=9))
from pathlib import Path
from runtime import data_dir, runs_dir

RUNS = str(runs_dir() / 'anchor5_clarifybench')


def agg_subset(subset, data):
    outdir = f"{RUNS}/results_{subset}"
    total = len([f for f in glob.glob(f"{data}/ClarifyBench_{subset}/*.json") if "RESULT" not in f])
    files = glob.glob(f"{outdir}/*_RESULT.json")
    task_success=[]; tool_match=[]; param_match=[]; exact=[]; exec_succ=[]; cov=[]
    n_eval=0; n_runerr=0
    for f in files:
        try: d=json.load(open(f))
        except Exception: n_runerr+=1; continue
        ev=d.get("evaluation")
        if not ev: n_runerr+=1; continue
        n_eval+=1
        corr=ev.get("correctness",{}); ex=ev.get("execution",{}); cv=ev.get("coverage",{})
        task_success.append(1.0 if ev.get("success") else 0.0)
        tool_match.append(corr.get("tool_match_rate",0.0))
        param_match.append(corr.get("param_match_rate",0.0))
        exact.append(1.0 if corr.get("exact_match") else 0.0)
        exec_succ.append(ex.get("success_rate",0.0))
        cov.append(cv.get("coverage_rate",0.0))
    def m(x): return round(float(np.mean(x))*100,2) if x else None
    return {"subset":subset,"n_total":total,"n_done":len(files),"n_evaluated":n_eval,"n_runerr":n_runerr,
            "task_success_pct":m(task_success),"tool_match_rate_pct":m(tool_match),
            "param_match_rate_pct":m(param_match),"exact_match_pct":m(exact),
            "execution_success_rate_pct":m(exec_succ),"coverage_rate_pct":m(cov)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default=os.environ.get("CLARIFYBENCH_ROOT"))
    args = parser.parse_args()
    if not args.benchmark_root:
        parser.error("Set --benchmark-root or CLARIFYBENCH_ROOT")
    data = Path(args.benchmark_root).expanduser().resolve() / "ClarifyBench"
    if not data.is_dir():
        parser.error(f"Expected the upstream data directory at {data}")
    Path(RUNS).mkdir(parents=True, exist_ok=True)
    res={s:agg_subset(s, data) for s in ["A","E","I"]}
    # 总体 (加权合并 evaluated)
    allf=[]
    for s in ["A","E","I"]: allf+=glob.glob(f"{RUNS}/results_{s}/*_RESULT.json")
    ts=[];tm=[];pm=[];ex=[];es=[];cv=[]
    for f in allf:
        try:d=json.load(open(f))
        except:continue
        e=d.get("evaluation")
        if not e:continue
        c=e.get("correctness",{});x=e.get("execution",{});v=e.get("coverage",{})
        ts.append(1.0 if e.get("success") else 0.0);tm.append(c.get("tool_match_rate",0.0))
        pm.append(c.get("param_match_rate",0.0));ex.append(1.0 if c.get("exact_match") else 0.0)
        es.append(x.get("success_rate",0.0));cv.append(v.get("coverage_rate",0.0))
    def m(a):return round(float(np.mean(a))*100,2) if a else None
    overall={"n_evaluated":len(ts),"task_success_pct":m(ts),"tool_match_rate_pct":m(tm),
             "param_match_rate_pct":m(pm),"exact_match_pct":m(ex),
             "execution_success_rate_pct":m(es),"coverage_rate_pct":m(cv)}
    out={"benchmark":"ClarifyBench","protocol":"Native","agent":"baseline (Autonomous, gpt-5.6-sol)",
         "anchor":"H0.anchor_5","timestamp":datetime.now(JST).isoformat(),
         "per_subset":res,"overall":overall,
         "note":"Native Protocol 官方 evaluator; A=Explicit E=Ambiguous I=Infeasible; Choice Adaptation 另行"}
    with open(f"{RUNS}/clarifybench_native_summary.json","w",encoding="utf-8") as fp:
        json.dump(out,fp,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
