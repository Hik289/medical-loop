"""Subprocess entry point: run ONE (benchmark, method, protocol, task) and write its JSON record.

python -m e1.worker --cfg resolved.json --benchmark clarifybench --method sage --protocol choice \
    --task_id E/003d59e9 --out path.json [--forced '{...}'] [--replay_cache dir]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from . import __version__
from .llm import LLMCallError, LLMClient, Ledger
from .methods import ADAPTATION_NOTES
from .store import write_json

HUMAN_TIME_NOTE = ("human_seconds is null: answers come from the benchmark's simulated user / judge, so no "
                   "human time exists to measure. It is NOT estimated. Real-human runs must log it.")
PROTOCOL_NOTES = {
    "native": "Native protocol: questions are asked in free text and answered by the benchmark's own "
              "simulated user (ClarifyBench UserSimulator) or ask_human judge (HiL-SQL).",
    "choice": "Choice adaptation (HumanLoop §11): each question is a local multiple-choice item with 2-3 "
              "substantive options + Insufficient/None/Skip (shuffled, no recommendation); the simulated user "
              "picks one option using the same role information. Not an official leaderboard protocol.",
}
RESPONDENT_NOTES = {
    "official_sim": "Clarifications answered by ClarifyBench's official UserSimulator (LLM). Its prompt contains "
                    "the task's full user_intention, turn-1 ground-truth call(s) and context; it paraphrases.",
    "gold_intention": "Clarifications answered by the task's scripted clarification for the current request, "
                      "copied verbatim from user_intention (NoisyToolBench gold-clarification protocol; no LLM, "
                      "ground_truth_tool_calls never read). If the request has no scripted clarification the reply "
                      "is a fixed no-information sentence (r_type=insufficient). The agent's tool calls are still "
                      "re-derived by the real action LLM call and scored by the official evaluator.",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forced", default="")
    ap.add_argument("--replay_cache", default="")
    a = ap.parse_args(argv)
    with open(a.cfg, encoding="utf-8") as f:
        cfg = json.load(f)
    forced = json.loads(a.forced) if a.forced else None
    ledger = Ledger()
    t0 = time.time()
    rec = {"task_id": a.task_id, "benchmark": a.benchmark, "method": a.method, "protocol": a.protocol,
           "split": a.split, "forced_branch": forced, "status": "error", "error": None,
           "started": datetime.now(timezone.utc).isoformat(), "code_version": __version__,
           "config_hash": cfg.get("_config_hash"), "adaptation": ADAPTATION_NOTES.get(a.method),
           "protocol_note": PROTOCOL_NOTES[a.protocol], "human_seconds": None, "human_time_note": HUMAN_TIME_NOTE,
           "pid": os.getpid()}
    try:
        llm = LLMClient(cfg, ledger, replay_cache_dir=a.replay_cache or None)
        rec["model"] = llm.meta()
        if a.benchmark == "clarifybench":
            from . import cb_adapter
            tasks = cb_adapter.list_tasks(cfg)
            res = cb_adapter.run_task(cfg, tasks[a.task_id], a.method, a.protocol, llm, forced)
        elif a.benchmark == "hilsql":
            from . import hilsql
            task = hilsql.list_tasks(cfg)[a.task_id]
            if not task["available"]:
                raise RuntimeError(f"task unavailable: {task.get('reason')}")
            res = hilsql.run_task(cfg, a.task_id, task, a.method, a.protocol, llm, forced)
        else:
            raise ValueError(a.benchmark)
        rec.update(res)
        if "respondent" in res:
            rec["respondent_note"] = RESPONDENT_NOTES[res["respondent"]]
        api = ledger.summary(cfg["llm"].get("price_per_1m_input"), cfg["llm"].get("price_per_1m_output"))
        if api["n_errors"] and cfg["run"].get("strict_llm_errors", True):
            rec["status"] = "error"
            rec["error"] = f"{api['n_errors']} failed API call(s); see api.errors"
        elif res.get("metrics") is None or res.get("Y") is None:
            rec["status"] = "error"
            rec["error"] = "official evaluator returned no score"
        else:
            rec["status"] = "measured"
    except LLMCallError as e:
        rec["error"] = f"LLMCallError: {e}"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    rec["api"] = ledger.summary(cfg["llm"].get("price_per_1m_input"), cfg["llm"].get("price_per_1m_output"))
    rec["wall_s"] = round(time.time() - t0, 2)
    rec["finished"] = datetime.now(timezone.utc).isoformat()
    write_json(a.out, rec)
    return 0 if rec["status"] == "measured" else 2


if __name__ == "__main__":
    sys.exit(main())
