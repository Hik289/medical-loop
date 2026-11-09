"""Orchestrator: plans jobs, resumes from per-task caches, runs <=10 worker subprocesses with a
hard per-task timeout (process-group kill), and freezes the ClarifyBench split."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import MAX_CONCURRENCY, validate
from .store import cached_status, read_json, sha256_text, task_path, write_json

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolved_cfg_path(cfg: Dict[str, Any]) -> str:
    out = cfg["run"]["out_dir"]
    os.makedirs(out, exist_ok=True)
    body = json.dumps({k: v for k, v in cfg.items() if k != "_config_hash"}, sort_keys=True)
    cfg["_config_hash"] = sha256_text(body)[:12]
    p = os.path.join(out, "configs", f"resolved_{cfg['_config_hash']}.json")
    write_json(p, cfg)
    return p


def task_universe(cfg: Dict[str, Any], benchmark: str) -> Dict[str, Dict[str, Any]]:
    """task_id -> {"split": ..., "available": bool}; ClarifyBench split frozen on first use."""
    if benchmark == "clarifybench":
        from .cb_adapter import list_tasks, split_of
        sp = cfg["split"]["clarifybench"]
        tasks = {t: {"split": split_of(t, int(sp["train_pct"]), int(sp["dev_pct"])), "available": True}
                 for t in list_tasks(cfg)}
        path = os.path.join(cfg["run"]["out_dir"], "splits", "clarifybench_split.json")
        cur = {"train_pct": sp["train_pct"], "dev_pct": sp["dev_pct"], "split": {t: v["split"] for t, v in tasks.items()}}
        old = read_json(path)
        if old is None:
            write_json(path, cur)
        elif old != cur:
            raise RuntimeError(f"ClarifyBench split differs from frozen {path}; refuse to continue")
        return tasks
    from .hilsql import list_tasks
    # Only 13 'financial' tasks are executable locally; all are evaluation-only (no HiL-SQL training).
    tasks = list_tasks(cfg)
    write_json(os.path.join(cfg["run"]["out_dir"], "splits", "hilsql_availability.json"),
               {t: {"db": v["db"], "available": v["available"], "reason": v.get("reason")} for t, v in tasks.items()})
    return {t: {"split": "test", "available": v["available"], "db": v["db"], "reason": v.get("reason"),
                "db_path": (v.get("dbcfg") or {}).get("sqlite")}
            for t, v in tasks.items()}


def _run_one(cfg_path: str, cfg: Dict[str, Any], job: Dict[str, Any]) -> str:
    out = job["out"]
    cmd = [cfg["run"].get("python") or sys.executable, "-m", "e1.worker", "--cfg", cfg_path,
           "--benchmark", job["benchmark"], "--method", job["method"], "--protocol", job["protocol"],
           "--task_id", job["task_id"], "--split", job.get("split", ""), "--out", out]
    if job.get("forced"):
        cmd += ["--forced", json.dumps(job["forced"])]
    if job.get("replay_cache"):
        cmd += ["--replay_cache", job["replay_cache"]]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    logf = out[:-5] + ".log"
    env = dict(os.environ, PYTHONPATH=PKG_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""),
               E1_LOG_DIR=os.path.dirname(out))
    timeout = int(cfg["run"]["task_timeout_s"])
    t0 = time.time()
    with open(logf, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=PKG_DIR, env=env, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            write_json(out, {"task_id": job["task_id"], "benchmark": job["benchmark"], "method": job["method"],
                             "protocol": job["protocol"], "split": job.get("split"), "respondent": job.get("respondent"), "forced_branch": job.get("forced"),
                             "status": "timeout", "error": f"exceeded task_timeout_s={timeout}",
                             "wall_s": round(time.time() - t0, 2), "config_hash": cfg.get("_config_hash"),
                             "finished": datetime.now(timezone.utc).isoformat()})
            return "timeout"
    st = cached_status(out)
    if st is None:  # worker crashed before writing
        write_json(out, {"task_id": job["task_id"], "benchmark": job["benchmark"], "method": job["method"],
                         "protocol": job["protocol"], "split": job.get("split"), "respondent": job.get("respondent"), "status": "error",
                         "error": f"worker exited {p.returncode} without a record; see {logf}",
                         "config_hash": cfg.get("_config_hash")})
        return "error"
    return st


def run_jobs(cfg: Dict[str, Any], jobs: List[Dict[str, Any]]) -> Dict[str, int]:
    validate(cfg)
    conc = min(int(cfg["run"]["concurrency"]), MAX_CONCURRENCY)
    cfg_path = resolved_cfg_path(cfg)
    todo, counts = [], {"measured": 0, "error": 0, "timeout": 0, "skipped_cached": 0}
    for j in jobs:
        st = cached_status(j["out"])
        if st == "measured" or (st in ("error", "timeout") and not cfg["run"].get("retry_errors")):
            counts["skipped_cached"] += 1
            continue
        todo.append(j)
    log(f"{len(jobs)} jobs: {len(todo)} to run, {counts['skipped_cached']} cached; concurrency={conc}")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(_run_one, cfg_path, cfg, j): j for j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                st = f.result()
            except Exception as e:  # noqa: BLE001
                st = "error"
                log(f"orchestrator error {j['task_id']}: {e}")
            with lock:
                counts[st] = counts.get(st, 0) + 1
            if i % 10 == 0 or i == len(todo) or st != "measured":
                log(f"  [{i}/{len(todo)}] {j['benchmark']}/{j['protocol']}/{j['method']} {j['task_id']} -> {st}")
    log(f"done: {counts}")
    return counts


def plan(cfg: Dict[str, Any], benchmark: str, methods: List[str], protocols: List[str], split: str,
         limit: int = 0, task_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    uni = task_universe(cfg, benchmark)
    ids = [t for t, v in sorted(uni.items()) if v["available"] and (split == "all" or v["split"] == split)]
    if task_ids:
        ids = [t for t in ids if t in set(task_ids)]
    if limit:
        ids = ids[:limit]
    resp = respondent_tag(cfg, benchmark)
    jobs = []
    for m in methods:
        for p in protocols:
            if m in ("autonomous", "compute_matched") and p == "choice":
                continue  # no human channel -> protocol-independent; reported once under 'native'
            if resp and m in ("autonomous", "compute_matched"):
                continue  # no human channel -> respondent-independent; reference stays native/<m>
            for t in ids:
                j = {"benchmark": benchmark, "method": m, "protocol": p, "task_id": t, "split": uni[t]["split"],
                     "out": task_path(cfg["run"]["out_dir"], benchmark, p, m, t, tag=resp)}
                if resp:
                    j["respondent"] = resp
                jobs.append(j)
    return jobs


def respondent_tag(cfg: Dict[str, Any], benchmark: str) -> str:
    """'' for the default official simulator (cache layout unchanged), else the respondent name; records of a
    non-default respondent live in <protocol>/<method>@<respondent>/ so they never mix with the baseline."""
    if benchmark != "clarifybench":
        return ""
    r = (cfg.get("clarifybench") or {}).get("respondent") or "official_sim"
    return "" if r == "official_sim" else r
