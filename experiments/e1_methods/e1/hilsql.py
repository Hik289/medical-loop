"""HiL-Bench SQL adaptation (labelled): the official harness needs docker MCP servers +
SWE-agent + chromadb, which are not required by this runtime. We keep, from the official release:
  * the task instruction, database, business documentation and blocker registry,
  * the ask_human judge prompt (imported from hil_bench/ask_human_server.py) and its
    answer rules (resolution / "irrelevant question" / per-blocker cap of 2),
  * the official verifier functions (tests/test_verify.py: execute_sql, compare_pandas_outputs)
    run against ground_truth.sql exactly like test_verify.main(),
  * the official ask metrics (precision / recall / ask-F1 from compute_hil_metrics).
Adapted: the tool shell is re-implemented in-process (same commands), and
get_business_info uses lexical top-k retrieval instead of the chromadb semantic index.
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time
import types
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .controller import Controller
from .llm import LLMClient, parse_json
from .questions import choice_reply_text, parse_choice

CANT_ANSWER = "can't answer (perhaps transient hiccup)"
IRRELEVANT_QUESTION = "irrelevant question"


# ---------------------------------------------------------------- official modules
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def load_official_judge(hil_root: str):
    """Import hil_bench/ask_human_server.py with flask/litellm stubbed (only the prompt builder
    and constants are used; the HTTP server and litellm backend are not)."""
    for stub in ("litellm", "flask"):
        if stub not in sys.modules:
            m = types.ModuleType(stub)
            if stub == "flask":
                class _F:  # minimal Flask stand-in for decorators at import time
                    def __init__(self, *a, **k): pass

                    def route(self, *a, **k):
                        return lambda f: f
                m.Flask, m.jsonify, m.request = _F, (lambda *a, **k: None), None
            sys.modules[stub] = m
    return _load_module("e1_official_ask_human_server", os.path.join(hil_root, "hil_bench", "ask_human_server.py"))


def compute_hil_metrics(log: Dict[str, Any]) -> Dict[str, float]:
    """Per-instance formula of hil_bench/utils/compute_hil_metrics.py."""
    qs = log.get("questions") or []
    n_disc = len({e["blocker_name"] for e in qs if e.get("blocker_name") is not None})
    prec = n_disc / len(qs) if qs else 0.0
    rec = n_disc / log["n_blockers"] if log["n_blockers"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "ask_f1": f1, "n_questions": len(qs), "n_blockers": log["n_blockers"],
            "n_blockers_discovered": n_disc}


class Judge:
    def __init__(self, official_mod, registry_path: str, llm: LLMClient):
        with open(registry_path, encoding="utf-8") as f:
            reg = official_mod.BlockerRegistry(**json.load(f)).to_dict()
        self.mod, self.reg, self.llm = official_mod, reg, llm
        # bypass AskHuman.__init__ (it configures litellm); reuse the prompt builder verbatim
        self.ah = official_mod.AskHuman.__new__(official_mod.AskHuman)
        self.ah.blocker_registry = reg
        self.cap = int(getattr(official_mod, "MAX_ANSWERED_QUESTIONS_PER_BLOCKER", 2))
        self.log = {"questions": [], "n_blockers": len(reg), "blockers": {k: False for k in reg}}
        self.errors = 0

    def match(self, question: str) -> Optional[str]:
        prompt = self.ah._create_comprehensive_prompt(question)
        try:
            o = parse_json(self.llm.chat([{"role": "user", "content": prompt}], "judge:ask_human_match"))
        except Exception:
            self.errors += 1
            return "__ERROR__"
        k = o.get("blocker_key")
        return k if isinstance(k, str) and k in self.reg else None

    def ask(self, question: str) -> Tuple[str, Optional[str]]:
        k = self.match(question)
        if k == "__ERROR__":
            resp, k = CANT_ANSWER, None
        elif k is None:
            resp = IRRELEVANT_QUESTION
        else:
            prior = sum(1 for e in self.log["questions"] if e["blocker_name"] == k)
            if prior >= self.cap:
                resp, k = IRRELEVANT_QUESTION, None
            else:
                resp = self.reg[k].resolution
                self.log["blockers"][k] = True
        self.log["questions"].append({"question": question, "response": resp, "blocker_name": k})
        return resp, k


# ---------------------------------------------------------------- environment
class SQLEnv:
    def __init__(self, db_name: str, dbcfg: Dict[str, str], row_cap: int, sql_timeout_s: int, top_k: int):
        self.db_name, self.db = db_name, dbcfg["sqlite"]
        self.row_cap, self.sql_timeout_s, self.top_k = row_cap, sql_timeout_s, top_k
        self.desc_dir = dbcfg.get("description_dir")
        self.biz: List[str] = []
        if dbcfg.get("business_info") and os.path.exists(dbcfg["business_info"]):
            with open(dbcfg["business_info"], encoding="utf-8") as f:
                self.biz = [str(x) for x in json.load(f).get("business_info", [])]

    def _conn(self):
        return sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)

    def tables(self) -> List[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    def _desc_rows(self, fname: str) -> List[Dict[str, str]]:
        p = os.path.join(self.desc_dir or "", fname)
        if not self.desc_dir or not os.path.exists(p):
            return []
        with open(p, encoding="utf-8", errors="replace") as f:
            return list(csv.DictReader(f))

    def get_database_info(self, _arg: str = "") -> str:
        td = {r.get("table_name"): r.get("table_description", "") for r in self._desc_rows("table_descriptions.csv")}
        return "\n".join(f"- {t}: {td.get(t, '')}" for t in self.tables())

    def get_table_info(self, table: str) -> str:
        table = table.strip().strip("'\"")
        if table not in self.tables():
            return f"Error: unknown table {table}"
        with self._conn() as c:
            cols = c.execute(f"PRAGMA table_info('{table}')").fetchall()
        desc = {(r.get("original_column_name") or r.get("column_name") or "").strip(): r for r in self._desc_rows(f"{table}.csv")}
        lines = []
        for col in cols:
            d = desc.get(col[1], {})
            extra = "; ".join(f"{k}={v}" for k, v in d.items() if v and k not in ("original_column_name",))
            lines.append(f"- {col[1]} ({col[2]}){': ' + extra if extra else ''}")
        return "\n".join(lines)

    def get_column_info(self, arg: str) -> str:
        parts = arg.replace(",", " ").split()
        if len(parts) < 2:
            return "Error: usage get_column_info <table> <column>"
        info = self.get_table_info(parts[0])
        hit = [l for l in info.splitlines() if l.startswith(f"- {parts[1]} ")]
        return hit[0] if hit else f"Error: unknown column {parts[1]} in {parts[0]}"

    def get_business_info(self, query: str) -> str:
        if not self.biz:
            return "No business documentation available."
        tok = lambda s: re.findall(r"[a-z0-9_]+", s.lower())  # noqa: E731
        docs = [tok(d) for d in self.biz]
        df = Counter(w for d in docs for w in set(d))
        q = tok(query)
        n = len(docs)
        scores = []
        for i, d in enumerate(docs):
            tf = Counter(d)
            s = sum(tf[w] * math.log(1 + n / (1 + df[w])) for w in q if w in tf)
            scores.append((s, i))
        best = [i for s, i in sorted(scores, reverse=True)[: self.top_k] if s > 0]
        return "\n".join(f"- {self.biz[i]}" for i in best) or "No relevant documentation found."

    def execute_sql(self, query: str) -> str:
        if not query.strip():
            return "Error: empty query"
        try:
            with self._conn() as c:
                t0 = time.time()
                c.set_progress_handler(lambda: 1 if time.time() - t0 > self.sql_timeout_s else 0, 1000)
                cur = c.execute(query)
                if cur.description is None:
                    return "Query executed (no result set)."
                cols = [d[0] for d in cur.description]
                rows = cur.fetchmany(self.row_cap + 1)
            more = len(rows) > self.row_cap
            body = "\n".join(" | ".join(str(v) for v in r) for r in rows[: self.row_cap])
            return f"{' | '.join(cols)}\n{body}" + (f"\n... (truncated at {self.row_cap} rows)" if more else "")
        except Exception as e:  # noqa: BLE001 - SQL errors are observations
            return f"Error: {e}"


# ---------------------------------------------------------------- agent loop
SYS = """You are an expert SQL analyst agent. Solve the task by issuing ONE command per turn.
Commands (JSON output only):
  {{"thought": str, "command": "get_database_info", "arg": ""}}
  {{"thought": str, "command": "get_table_info", "arg": "<table>"}}
  {{"thought": str, "command": "get_column_info", "arg": "<table> <column>"}}
  {{"thought": str, "command": "get_business_info", "arg": "<question>"}}
  {{"thought": str, "command": "execute_sql", "arg": "<SQLite query>"}}
  {{"thought": str, "command": "submit_sql", "arg": "<final SQLite query>"}}{ask}
Submitting ends the task; if you never submit, the task scores 0."""
ASK_CMD = """
  {"thought": str, "command": "ask_human", "arg": "<one concise question>"}  (a human who knows the intended
   definitions answers; ask only about genuinely missing or ambiguous information)"""

CHOICE_USER = """You are the stakeholder who requested this SQL analysis. You know exactly this piece of
information about the request:
{info}
The analyst asks you a multiple-choice question:
{mc}
Pick the ONE option id consistent with what you know. Choose "none" if no substantive option is consistent
with it. Respond with JSON only: {{"choice": "<option id>"}}"""


def task_db(cfg: Dict[str, Any], tid: str) -> Optional[Dict[str, str]]:
    """Per-task /data extracted from the task's own docker image (scripts/extract_hilsql_dbs.sh)."""
    root = cfg["hilsql"].get("task_db_root")
    if not root:
        return None
    d = os.path.join(root, tid, "data")
    if not os.path.exists(os.path.join(d, "database.sqlite")):
        return None
    return {"sqlite": os.path.join(d, "database.sqlite"), "business_info": os.path.join(d, "business_info.json"),
            "description_dir": os.path.join(d, "database_description")}


def list_tasks(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    root = os.path.join(cfg["hilsql"]["root"], cfg["hilsql"]["tasks_subdir"])
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "hil-bench-sql_*-ask_human"))):
        with open(os.path.join(d, "task.toml"), encoding="utf-8") as f:
            m = re.search(r'database_name\s*=\s*"([^"]+)"', f.read())
        db = m.group(1) if m else None
        tid = os.path.basename(d).replace("-ask_human", "")
        dbcfg = task_db(cfg, tid) or (cfg["hilsql"].get("databases") or {}).get(db)
        if not dbcfg:
            out[tid] = {"dir": d, "db": db, "available": False, "reason": f"database {db!r} not configured"}
            continue
        pf = preflight(d, dbcfg["sqlite"])
        out[tid] = {"dir": d, "db": db, "dbcfg": dbcfg, "available": pf["ok"], "reason": pf["reason"]}
    return out


def preflight(task_dir: str, db_path: str) -> Dict[str, Any]:
    """Environment validity check (official verifier semantics): ground_truth.sql must execute on the
    configured DB and its output must match tests/golden_output.csv. Tasks failing this are NOT run
    (the DB is not the task's environment) and are reported as unavailable."""
    tv = _load_module(f"e1_tv_pf_{abs(hash(task_dir))}", os.path.join(task_dir, "tests", "test_verify.py"))
    with open(os.path.join(task_dir, "tests", "ground_truth.sql"), encoding="utf-8") as f:
        err, gt_df = tv.execute_sql(db_path, f.read().strip())
    if gt_df is None:
        return {"ok": False, "reason": f"ground_truth.sql fails on configured DB: {err}"}
    gp = os.path.join(task_dir, "tests", "golden_output.csv")
    if os.path.exists(gp) and hasattr(tv, "parse_golden_output"):
        with open(gp, encoding="utf-8") as f:
            golden = tv.parse_golden_output(f.read())
        if not tv.compare_pandas_outputs(golden, gt_df):
            return {"ok": False, "reason": "ground_truth output on configured DB != golden_output.csv"}
    return {"ok": True, "reason": None}


def score_submission(task_dir: str, db_path: str, sql: Optional[str]) -> Dict[str, Any]:
    """Exactly test_verify.main(): run submission and ground_truth.sql, compare_pandas_outputs(gt, sub)."""
    tv = _load_module(f"e1_tv_{abs(hash(task_dir))}", os.path.join(task_dir, "tests", "test_verify.py"))
    if not sql:
        return {"solve": 0, "submitted": False}
    with open(os.path.join(task_dir, "tests", "ground_truth.sql"), encoding="utf-8") as f:
        gt_sql = f.read().strip()
    sub_err, sub_df = tv.execute_sql(db_path, sql)
    gt_err, gt_df = tv.execute_sql(db_path, gt_sql)
    solve = 0 if (sub_df is None or gt_df is None) else int(tv.compare_pandas_outputs(gt_df, sub_df))
    return {"solve": solve, "submitted": True, "submission_error": sub_err, "gt_error": gt_err,
            "pred_rows": None if sub_df is None else int(len(sub_df)),
            "gt_rows": None if gt_df is None else int(len(gt_df))}


def run_task(cfg: Dict[str, Any], task_id: str, task: Dict[str, Any], method: str, protocol: str,
             llm: LLMClient, forced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    hcfg, b = cfg["hilsql"], cfg["budget"]["hilsql"]
    dbcfg = task["dbcfg"]
    env = SQLEnv(task["db"], dbcfg, int(hcfg["row_cap"]), int(hcfg["sql_timeout_s"]), int(hcfg["business_top_k"]))
    judge = Judge(load_official_judge(hcfg["root"]),
                  os.path.join(task["dir"], "environment", "ask-human-data", "blocker_registry.json"), llm)
    seed = int(re.sub(r"\D", "", task_id) or 0)
    ctrl = Controller(cfg, method, protocol, "hilsql", llm, int(b["max_checkpoints"]), int(b["max_questions"]),
                      seed, forced=forced, question_scope="task")
    ctrl.new_request()
    with open(os.path.join(task["dir"], "instruction.md"), encoding="utf-8") as f:
        instruction = f.read().split("# Available Commands")[0].strip()
    native = method == "native_ask_human"
    sys_p = SYS.format(ask=ASK_CMD if native else "")  # budget enforced at call time (observation on overuse)
    tools_text = "get_database_info | get_table_info | get_column_info | get_business_info | execute_sql | submit_sql"
    history: List[Dict[str, str]] = []
    observations: List[str] = []
    clarifications: List[str] = []
    submitted, n_err, steps = None, 0, 0

    def request_text():
        if not clarifications:
            return instruction
        return instruction + "\n\nClarifications from the stakeholder:\n" + "\n".join(clarifications)

    def human(text_q: str, pend: Dict[str, Any]) -> str:
        if protocol == "native":
            resp, _k = judge.ask(text_q)
            ctrl.on_reply({"r_type": "free_text", "text": resp})
            return resp
        q = pend["q"]
        k = judge.match(q["question"])
        if k == "__ERROR__":
            judge.log["questions"].append({"question": q["question"], "response": CANT_ANSWER, "blocker_name": None})
            parsed = {"choice_id": "insufficient", "r_type": "insufficient", "text": "Insufficient Evidence",
                      "judge_error": True}
        elif k is None or sum(1 for e in judge.log["questions"] if e["blocker_name"] == k) >= judge.cap:
            judge.log["questions"].append({"question": q["question"], "response": IRRELEVANT_QUESTION, "blocker_name": None})
            parsed = {"choice_id": "insufficient", "r_type": "insufficient", "text": "Insufficient Evidence",
                      "note": "question matched no blocker: the stakeholder has no information on it"}
        else:
            bl = judge.reg[k]
            judge.log["questions"].append({"question": q["question"], "response": bl.resolution, "blocker_name": k})
            judge.log["blockers"][k] = True
            raw = llm.json("", CHOICE_USER.format(info=f"{bl.description}\nResolution: {bl.resolution}",
                                                  mc=pend["rendered"]), "user_sim:choice")
            parsed = parse_choice(raw, q)
        shown = choice_reply_text(parsed)
        ctrl.on_reply(dict(parsed, text=parsed["text"] if parsed["r_type"] == "substantive" else shown))
        return shown

    while steps < int(b["max_steps"]):
        steps += 1
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": f"TASK:\n{request_text()}"}] + history
        o = parse_json(llm.chat(msgs, "agent:step"))
        cmd, arg = str(o.get("command", "")), o.get("arg", "")
        arg = arg if isinstance(arg, str) else json.dumps(arg)
        a0 = {"command": cmd, "arg": arg}
        if native and cmd == "ask_human" and not ctrl.native_may_ask():
            obs = "Error: ask_human unavailable (question budget exhausted)"
            n_err += 1
        elif native and cmd == "ask_human":
            pend = ctrl.native_question(arg, {"request": request_text(), "tools_text": tools_text,
                                              "observations": observations, "proposed": {}})
            obs = human(pend["rendered"], pend) if pend else "Error: question could not be posed"
            clarifications.append(f"Q: {arg}\nA: {obs}")
        else:
            state = {"request": request_text(), "tools_text": tools_text, "observations": list(observations),
                     "proposed": a0, "step_idx": steps, "n_tool_errors": n_err}
            out = ctrl.on_proposal(state, is_checkpoint=cmd in ("execute_sql", "submit_sql"))
            if out["type"] == "ask":
                pend = ctrl.pending
                ans = human(out["text"], pend)
                clarifications.append(f"Q: {pend['q']['question']}\nA: {ans}")
                obs = f"[asked stakeholder] {out['text']}\n[answer] {ans}"
                history += [{"role": "assistant", "content": json.dumps(o, ensure_ascii=False)},
                            {"role": "user", "content": "OBSERVATION:\n" + obs + "\nRe-plan your next command with this answer."}]
                observations.append(obs[:2000])
                continue
            a = out["action"] if isinstance(out["action"], dict) else a0
            cmd, arg = a.get("command", ""), a.get("arg", "")
            if cmd == "submit_sql":
                submitted = arg
                history.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
                break
            fn = getattr(env, cmd, None) if cmd in ("get_database_info", "get_table_info", "get_column_info",
                                                     "get_business_info", "execute_sql") else None
            obs = fn(arg) if fn else f"Error: unknown command {cmd!r}"
            if obs.startswith("Error"):
                n_err += 1
            o = dict(o, command=cmd, arg=arg)
        history += [{"role": "assistant", "content": json.dumps(o, ensure_ascii=False)},
                    {"role": "user", "content": "OBSERVATION:\n" + str(obs)[:6000]}]
        observations.append(f"{cmd}: {str(obs)[:600]}")
    sc = score_submission(task["dir"], dbcfg["sqlite"], submitted)
    hm = compute_hil_metrics(judge.log)
    return {"metrics": {"solve": float(sc["solve"]), "ask_precision": hm["precision"], "ask_recall": hm["recall"],
                        "ask_f1": hm["ask_f1"]},
            "Y": float(sc["solve"]), "score": sc, "hil_log": judge.log, "hil_metrics": hm,
            "judge_errors": judge.errors, "steps": steps, "submitted_sql": submitted,
            "n_user_questions": hm["n_questions"], "controller": ctrl.summary(), "trace": ctrl.trace,
            "teacher": ctrl.teacher}
