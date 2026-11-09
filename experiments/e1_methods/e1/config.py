"""Portable benchmark configuration; access and paths are supplied locally."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("HUMANCHOICE_ROOT", str(Path(__file__).resolve().parents[3])))
from typing import Any, Dict

MAX_CONCURRENCY = 10  # hard cap requested for E1 runs

METHODS = [
    "autonomous",          # same base agent, no human channel
    "native_ask_human",    # official ClarifyBench baseline / free ask_human tool (control, §13.2)
    "compute_matched",     # Compute-Matched Autonomous Branch-and-Aggregate
    "clam_api",            # CLAM with verbalized ambiguity score instead of token logprob
    "awn",                 # Ask-when-Needed argument check before each tool call
    "uot",                 # Uncertainty of Thoughts (hypothesis set, IG, reward propagation)
    "atd",                 # Active Task Disambiguation (solution sampling + EIG partition)
    "sage",                # SAGE-Agent inference-time structured uncertainty + EVPI
    "voi",                 # VoI communication: E[U|ask] - U(no ask) - cost
    "humanchoice",         # Q -> G -> A trained MLPs (requires trained modules)
]
PROTOCOLS = ["native", "choice"]
BENCHMARKS = ["clarifybench", "hilsql"]

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "provider": "openai",              # "openai" or "fake:<path/to/responder.py>"
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("MODEL_NAME", "gpt-5.6-sol"),
        "api_key_env": "OPENAI_API_KEY",   # read from the local environment
        "reasoning_effort": "high",        # sent via extra_body; "" disables
        "request_timeout_s": 180,
        "max_retries": 6,                  # OpenAI SDK transport retries
        "app_retries": 2,                  # extra retries around the SDK on exceptions
        "price_per_1m_input": None,        # USD; cost reported only if both prices are set
        "price_per_1m_output": None,
    },
    "run": {
        "out_dir": str(PROJECT_ROOT / "runs/e1_methods"),
        "concurrency": 8,
        "task_timeout_s": 900,
        "strict_llm_errors": True,         # any failed API call => task status "error"
        "retry_errors": False,             # re-run tasks cached with status error/timeout
        "python": None,                    # interpreter for workers (default: sys.executable)
    },
    "budget": {
        # HumanLoop §2.1: <=3 decision checkpoints, one human opportunity.
        "clarifybench": {"max_checkpoints_per_request": 3, "max_questions_per_request": 1},
        "hilsql": {"max_checkpoints": 6, "max_questions": 3, "max_steps": 30},
        "enforce_on_native": True,  # native_ask_human gets the same question budget
    },
    "split": {
        # ClarifyBench has no official train/dev/test split -> frozen hash split by task id (§12.2)
        "clarifybench": {"train_pct": 30, "dev_pct": 10},
    },
    "methods": {
        "compute_matched": {"k": 4, "rounds": 1},
        "clam_api": {"tau": 0.5},
        "uot": {"n_hypotheses": 5, "n_questions": 4, "min_reward": 1e-6},
        "atd": {"n_candidates": 5, "n_questions": 4, "min_eig": 1e-6},
        "sage": {"tau": 0.1, "alpha_per_arg": 0.02, "redundancy_penalty": 1.0, "max_subset": 2},
        "voi": {"n_hypotheses": 5, "n_questions": 3, "comm_cost": 0.05},
        "humanchoice": {"modules_dir": None, "seed": 42, "max_pool": 3, "policy": "trained"},
    },
    "clarifybench": {
        "root": str(PROJECT_ROOT / "data/benchmarks/ClarifyBench"),
        "subsets": ["A", "E", "I"],
        "eval_split": "test",
        "y_metric": "correctness.param_match_rate",   # scalar Y for HumanChoice teacher labels
        # who answers clarification questions: "official_sim" (benchmark UserSimulator, default, unchanged)
        # | "gold_intention" (verbatim scripted clarification from user_intention; separate cache dir
        # <method>@gold_intention). See README "Gold-intention respondent".
        "respondent": "official_sim",
    },
    "hilsql": {
        "root": str(PROJECT_ROOT / "data/benchmarks/hil-bench"),
        "tasks_subdir": "harbor_sql",
        # db_name -> {"sqlite":..., "business_info":..., "description_dir":...}; DBs not listed are "unavailable"
        "databases": {
            "financial": {
                "sqlite": str(PROJECT_ROOT / "data/benchmarks/hilsql_native/database.sqlite"),
                "business_info": str(PROJECT_ROOT / "data/benchmarks/hilsql_native/data_dir/business_info.json"),
                "description_dir": str(PROJECT_ROOT / "data/benchmarks/hilsql_native/data_dir/database_description"),
            }
        },
        # Each HiL-SQL task has a separate database under its configured task root. If set, this database
        # (extracted by scripts/extract_hilsql_dbs.sh) takes precedence over `databases`.
        "task_db_root": str(PROJECT_ROOT / "data/benchmarks/hilsql_task_dbs"),
        "row_cap": 30,
        "sql_timeout_s": 60,
        "business_top_k": 5,
    },
}


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_update(cfg, json.load(f))
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    c = int(cfg["run"]["concurrency"])
    if c < 1 or c > MAX_CONCURRENCY:
        raise ValueError(f"run.concurrency must be in [1, {MAX_CONCURRENCY}], got {c}")
    if int(cfg["run"]["task_timeout_s"]) <= 0:
        raise ValueError("run.task_timeout_s must be positive")
    r = (cfg.get("clarifybench") or {}).get("respondent") or "official_sim"
    if r not in ("official_sim", "gold_intention"):
        raise ValueError(f"clarifybench.respondent must be official_sim|gold_intention, got {r!r}")


def api_key(cfg: Dict[str, Any]) -> str:
    return os.environ.get(cfg["llm"].get("api_key_env") or "OPENAI_API_KEY", "")


def set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
