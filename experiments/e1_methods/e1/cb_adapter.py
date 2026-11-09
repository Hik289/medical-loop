"""ClarifyBench integration.

Runs one task through the OFFICIAL harness loop (`main.run_simulation`): official plugins,
tool executor, UserSimulator role information and SimulationEvaluator. We only
  * replace the LLM provider with an accounting bridge to the configured endpoint,
  * subclass the official ReactAgent so the method controller decides when/what to ask,
  * wrap the UserSimulator for the Choice protocol (same role information, MC answer),
  * optionally replace the clarification answer by the task's scripted gold clarification
    (`clarifybench.respondent = "gold_intention"`, see `gold_clarifications`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Optional

from .controller import Controller
from .llm import LLMCallError, LLMClient
from .questions import choice_reply_text, parse_choice, render_choice

CB_METRICS = {
    "success": ("success",),
    "tool_match_rate": ("correctness", "tool_match_rate"),
    "param_match_rate": ("correctness", "param_match_rate"),
    "exact_match": ("correctness", "exact_match"),
    "execution_success_rate": ("execution", "success_rate"),
    "coverage_rate": ("coverage", "coverage_rate"),
}

REASON_PROMPT = """You are an AI assistant. Help with the user's request by calling appropriate tools.

REQUEST: {request}

OBSERVATIONS:
{obs_text}

AVAILABLE TOOLS:
{tools}

Choose a tool to call:
- Use regular tools to complete the task. Don't call tools redundantly.
- Use final_answer when done, to give an answer to the user.

Respond in JSON:
{{
    "reasoning": "Your thinking",
    "tool_call": {{
        "tool_name": "tool_name",
        "arguments": {{"arg": "value"}}
    }}
}}
"""  # official baseline prompt with the ask_user bullet removed (human channel owned by the method)

CHOICE_USER_PROMPT = """
You are simulating a user who is interacting with an AI assistant.

Original query: "{query}"

User's intent for the CURRENT request: {intent}

Information needed for the CURRENT request (do not reveal future intentions):
{gt}

Additional context:
{context}

The AI assistant asks you the following multiple-choice question:
{mc}

Pick the ONE option id that matches what you want. Choose "none" if no substantive option matches your
intent, "insufficient" if your information does not determine the answer, and "skip" only if you do not
want to answer. NEVER BREAK CHARACTER. Respond with JSON only: {{"choice": "<option id>"}}
"""


# ------------------------------------------------------------------ respondents
# official_sim   : the benchmark's own UserSimulator (LLM). NOTE: it is NOT uninformed - its prompt
#                  contains the task's full `user_intention` (incl. every "If asked, user clarifies ..."
#                  line), turn-1 ground-truth call(s) and `context`. It paraphrases and may withhold.
# gold_intention : deterministic oracle reply = the task's scripted clarification text for the CURRENT
#                  request, copied verbatim from `user_intention` (NoisyToolBench gold-clarification
#                  protocol). No LLM call, never reads ground_truth_tool_calls. Question-independent.
RESPONDENTS = ("official_sim", "gold_intention")
DEFAULT_RESPONDENT = "official_sim"
GOLD_NO_INFO_REPLY = "I don't have any additional details beyond my request."

_REQ_RE = re.compile(r"\buser\s+requests\b", re.I)
_CLAR_RE = re.compile(r"if\s+asked,?\s*(?:the\s+)?user\s+clarifies\b\s*(?:that\b)?\s*:?\s*", re.I)
# residue of the NEXT item left before "user requests": "\n3. ", "Thereafter, ", "\n - Finally, "
_TRAIL_RE = re.compile(r"(?:\n[\s\-*]*(?:\d+\.)?)?\s*(?:\b(?:thereafter|finally|then|next|afterwards?),)?\s*$", re.I)
_VACUOUS = {"details", "the details", "more details", "further details", "details as needed"}


def respondent_of(cfg: Dict[str, Any]) -> str:
    r = (cfg.get("clarifybench") or {}).get("respondent") or DEFAULT_RESPONDENT
    if r not in RESPONDENTS:
        raise ValueError(f"clarifybench.respondent must be one of {RESPONDENTS}, got {r!r}")
    return r


def _clean_clar(t: str) -> str:
    t = t.split("\n\n", 1)[0]
    t = _REQ_RE.split(t, 1)[0]
    return _TRAIL_RE.sub("", t).strip()


def gold_clarifications(user_intention: str, n_requests: int) -> Dict[str, Any]:
    """Split ClarifyBench `user_intention` into the scripted clarification(s) of each request.

    Segments start at each "User requests" (text before the first one belongs to request 0); a
    clarification ("If asked, user clarifies [that] ...") belongs to the segment it appears in.
    Placeholder clarifications ("... clarifies details.") carry no information and are dropped.
    If the number of segments != number of requests the mapping is unreliable: `aligned=False`
    and every request gets all clarifications (the official simulator sees all of them anyway).
    """
    text = user_intention or ""
    starts = [m.start() for m in _REQ_RE.finditer(text)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    segs = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    if len(segs) > 1 and not _REQ_RE.search(segs[0]):  # preamble -> merge into request 0
        segs = [segs[0] + segs[1]] + segs[2:]
    per_seg: list = []
    for seg in segs:
        out = []
        for m in _CLAR_RE.finditer(seg):
            c = _clean_clar(seg[m.end():])
            if c and c.lower().strip(" .\"'") not in _VACUOUS:
                out.append(c)
        per_seg.append(out)
    allc = [c for cs in per_seg for c in cs]
    aligned = len(segs) == n_requests
    per_request = per_seg if aligned else [list(allc) for _ in range(n_requests)]
    return {"aligned": aligned, "n_segments": len(segs), "n_requests": n_requests,
            "n_clarifications": len(allc), "per_request": per_request}


def gold_answer(gold: Dict[str, Any], request_idx: int) -> Dict[str, Any]:
    pr = gold["per_request"]
    cl = pr[request_idx] if 0 <= request_idx < len(pr) else []
    return {"text": " ".join(cl) if cl else GOLD_NO_INFO_REPLY, "gold_available": bool(cl),
            "aligned": gold["aligned"], "request_idx": request_idx}


def extract_metrics(evaluation: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out = {}
    for k, path in CB_METRICS.items():
        v: Any = evaluation
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        out[k] = None if v is None else float(v)
    return out


def get_metric(evaluation: Dict[str, Any], dotted: str) -> Optional[float]:
    v: Any = evaluation
    for p in dotted.split("."):
        v = v.get(p) if isinstance(v, dict) else None
    return None if v is None else float(v)


def _strip_ask_user(desc: str) -> str:
    return "\n\n".join(b for b in desc.split("\n\n") if not b.startswith("Tool: ask_user"))


def run_task(cfg: Dict[str, Any], task_file: str, method: str, protocol: str, llm: LLMClient,
             forced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    root = os.path.abspath(cfg["clarifybench"]["root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)  # official plugin configs are loaded relative to the repo root
    import main as cbmain  # noqa: E402  (official harness)
    import config as cbconfig  # noqa: E402
    from core.baseline_agent import ReactAgent  # noqa: E402
    from core.types import ToolCall  # noqa: E402
    from llm.provider import LLMProvider  # noqa: E402
    from simulation.data_loader import SimulationDataLoader  # noqa: E402
    from utils.logger import setup_logger  # noqa: E402

    if cbmain.logger is None:
        cbmain.logger = setup_logger(log_file=os.path.join(os.environ.get("E1_LOG_DIR", "/tmp"), "cb_harness.log"))

    class Bridge(LLMProvider):
        """Same request shape as the harness's patched OpenAI-compatible provider; failures are
        swallowed exactly like the official provider but recorded in the ledger (=> task error)."""

        def __init__(self, role: str):
            self.role, self.model_name = role, llm.model

        def generate_text(self, prompt, max_tokens=None, temperature=None):
            try:
                return llm.chat([{"role": "user", "content": prompt}], f"{self.role}:text").strip()
            except LLMCallError:
                return ""

        def generate_json(self, prompt, response_model, max_tokens=None, temperature=None):
            sys_p = ("You are a helpful assistant that always responds with valid JSON.\n"
                     "Your responses must be properly formatted JSON objects only, with no additional text.")
            try:
                txt = llm.chat([{"role": "system", "content": sys_p},
                                {"role": "user", "content": self.enhance_json_prompt(prompt, response_model)}],
                               f"{self.role}:json")
            except LLMCallError:
                return {}
            return self.safe_parse_json(txt, default={})

    tid = os.path.splitext(os.path.basename(task_file))[0]
    seed = int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 100000
    b = cfg["budget"]["clarifybench"]
    ctrl = Controller(cfg, method, protocol, "clarifybench", llm, int(b["max_checkpoints_per_request"]),
                      int(b["max_questions_per_request"]), seed, forced=forced, question_scope="request")
    agent_provider, user_provider = Bridge("agent"), Bridge("user_sim")
    native = method == "native_ask_human"
    respondent = respondent_of(cfg)
    gold_log: list = []

    class MethodAgent(ReactAgent):
        def start_new_request(self, request_text, request_type="initial"):
            ctrl.new_request()
            self._e1_steps = 0
            self._e1_errors = 0
            return super().start_new_request(request_text, request_type)

        def _state(self, request, observations, proposed):
            return {"request": request, "tools_text": self._tools_text(), "observations": list(observations),
                    "proposed": proposed, "step_idx": getattr(self, "_e1_steps", 0),
                    "n_tool_errors": getattr(self, "_e1_errors", 0),
                    "tool_names": [t.name for t in self.tool_registry.get_all_tools()]}

        def _tools_text(self):
            d = self.tool_registry.get_tool_descriptions()
            return d if (native and ctrl.native_may_ask()) else _strip_ask_user(d)

        def _base_reason(self, request, observations):
            obs_text = "\n".join(f"- {o}" for o in observations) if observations else "None"
            prompt = REASON_PROMPT.format(request=request, obs_text=obs_text, tools=self._tools_text())
            schema = {"reasoning": "string", "tool_call": {"tool_name": "string", "arguments": {}}}
            r = self._call_llm_with_tracking(prompt=prompt, interaction_type="reasoning",
                                             response_model=schema, max_tokens=2000)
            tc = r.get("tool_call", {}) or {}
            if tc.get("tool_name") == "ask_user":
                ctrl.counters["base_ask_suppressed"] += 1
                r = self._call_llm_with_tracking(
                    prompt=prompt + "\nNOTE: ask_user is NOT available. Choose a regular tool or final_answer.",
                    interaction_type="reasoning", response_model=schema, max_tokens=2000)
                tc = r.get("tool_call", {}) or {}
                if tc.get("tool_name") == "ask_user":
                    tc = {"tool_name": "final_answer", "arguments": {"answer": r.get("reasoning", "")}}
            return r.get("reasoning", ""), {"tool_name": tc.get("tool_name", "final_answer"),
                                            "arguments": tc.get("arguments", {}) or {}}

        def _reason(self, request, observations):
            self._e1_steps = getattr(self, "_e1_steps", 0) + 1
            if native and ctrl.native_may_ask():
                reasoning, atype, res = super()._reason(request, observations)
                if atype == "ask_question":
                    pend = ctrl.native_question(res, self._state(request, observations, {}))
                    if pend is not None:
                        return reasoning, atype, pend["rendered"]
                return reasoning, atype, res
            reasoning, a0 = self._base_reason(request, observations)
            if native:  # question budget exhausted: same base agent, ask_user removed
                tc = ToolCall(a0["tool_name"], a0["arguments"])
                return reasoning, ("final_answer" if tc.tool_name == "final_answer" else "tool_call"), tc
            is_cp = a0["tool_name"] not in ("final_answer", "ask_user")
            out = ctrl.on_proposal(self._state(request, observations, a0), is_cp)
            if out["type"] == "ask":
                return reasoning, "ask_question", out["text"]
            a = out["action"] if isinstance(out["action"], dict) else a0
            tc = ToolCall(a.get("tool_name", "final_answer"), a.get("arguments", {}) or {})
            return reasoning, ("final_answer" if tc.tool_name == "final_answer" else "tool_call"), tc

        def _handle_error(self, error_result, request, context):
            self._e1_errors = getattr(self, "_e1_errors", 0) + 1
            if native and ctrl.native_may_ask():
                return super()._handle_error(error_result, request, context)
            return {"needs_clarification": False}

    official_user_cls = cbmain.UserSimulator

    class E1User:
        def __init__(self, llm_provider, ground_truth, user_intent=None):
            self.off = official_user_cls(llm_provider=user_provider, ground_truth=ground_truth,
                                         user_intent=user_intent)

        def get_response_to_question(self, question):
            pend = ctrl.pending
            if pend is None and native and not self.off._is_follow_up_question(question):
                # official error-clarification question of the native agent: register it like ask_user
                pend = ctrl.native_question(question, {"request": self.off.original_query, "tools_text": "",
                                                       "observations": [], "proposed": {}})
                if pend is not None and protocol == "choice":
                    question = pend["rendered"]
            if pend is None or pend.get("rendered") != question:
                return self.off.get_response_to_question(question)  # e.g. "anything else?"
            if respondent == "gold_intention":
                g = gold_answer(gold, ctrl.request_idx)
                ctrl.on_reply({"r_type": "free_text" if g["gold_available"] else "insufficient",
                               "text": g["text"], "respondent": "gold_intention",
                               "gold_available": g["gold_available"], "gold_aligned": g["aligned"]})
                gold_log.append(dict(g, question=question))
                return g["text"]
            if protocol == "native":
                resp = self.off.get_response_to_question(question)
                ctrl.on_reply({"r_type": "free_text", "text": resp})
                return resp
            prompt = CHOICE_USER_PROMPT.format(query=self.off.original_query, intent=self.off.user_intent,
                                               gt=self.off._get_current_turn_ground_truth(),
                                               context=self.off.context, mc=question)
            raw = user_provider.generate_json(prompt, {"choice": "string"})
            parsed = parse_choice(raw, pend["q"])
            parsed["text_shown"] = choice_reply_text(parsed)
            ctrl.on_reply(dict(parsed, text=parsed["text"] if parsed["r_type"] == "substantive" else parsed["text_shown"]))
            return parsed["text_shown"]

    cbmain.initialize_llm_provider = lambda _cfg: agent_provider
    cbmain.UserSimulator = E1User
    try:
        loader = SimulationDataLoader(os.path.dirname(task_file))
        data = loader.load_simulation_data(task_file)
        if respondent == "gold_intention":  # same request list as main.run_simulation
            n_req = 1 + len(data.get("follow_ups", data.get("potential_follow_ups", [])) or [])
            gold = gold_clarifications(data.get("user_intention", ""), n_req)
        plugin = cbmain.determine_plugin_from_simulation_data(data)
        if not plugin:
            raise ValueError("cannot determine plugin")
        pm, reg, prov = cbmain.initialize_components(plugin, data)
        sim_cfg = dict(cbconfig.SIMULATION_CONFIG)
        result = cbmain.run_simulation(data, pm, reg, prov, sim_cfg, False, MethodAgent)
    finally:
        cbmain.UserSimulator = official_user_cls
    ev = result.get("evaluation") or {}
    extra: Dict[str, Any] = {"respondent": respondent}
    if respondent == "gold_intention":
        extra["gold_intention"] = {k: gold[k] for k in ("aligned", "n_segments", "n_requests", "n_clarifications")}
        extra["gold_intention"].update(replies=gold_log, n_replies=len(gold_log),
                                       n_replies_with_gold=sum(1 for g in gold_log if g["gold_available"]))
    return {
        **extra,
        "metrics": extract_metrics(ev),
        "Y": get_metric(ev, cfg["clarifybench"]["y_metric"]),
        "evaluation": ev,
        "n_user_questions": sum(1 for c in result.get("conversation", []) if c.get("type") == "clarification"),
        "final_tool_calls": result.get("final_tool_calls"),
        "conversation": result.get("conversation"),
        "controller": ctrl.summary(),
        "trace": ctrl.trace,
        "teacher": ctrl.teacher,
    }


def list_tasks(cfg: Dict[str, Any]) -> Dict[str, str]:
    """task_id -> file. task_id = '<subset>/<file id>'."""
    root = cfg["clarifybench"]["root"]
    out = {}
    for s in cfg["clarifybench"]["subsets"]:
        d = os.path.join(root, "ClarifyBench", f"ClarifyBench_{s}")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and "RESULT" not in f:
                out[f"{s}/{f[:-5]}"] = os.path.join(d, f)
    return out


def gold_coverage(cfg: Dict[str, Any], task_ids=None) -> Dict[str, Any]:
    """Dataset-only count (no API): how many tasks/requests carry a scripted, non-placeholder clarification.
    Bounds what the gold_intention respondent can reveal; tasks without one get a no-information reply."""
    out: Dict[str, Any] = {}
    for tid, f in sorted(list_tasks(cfg).items()):
        if task_ids is not None and tid not in task_ids:
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        n_req = 1 + len(d.get("follow_ups", d.get("potential_follow_ups", [])) or [])
        g = gold_clarifications(d.get("user_intention", ""), n_req)
        s = out.setdefault(tid.split("/")[0], {"n_tasks": 0, "n_aligned": 0, "n_tasks_with_clarification": 0,
                                               "n_requests": 0, "n_requests_with_clarification": 0})
        s["n_tasks"] += 1
        s["n_aligned"] += int(g["aligned"])
        s["n_tasks_with_clarification"] += int(g["n_clarifications"] > 0)
        s["n_requests"] += n_req
        s["n_requests_with_clarification"] += sum(1 for c in g["per_request"] if c)
    return out


def split_of(task_id: str, train_pct: int, dev_pct: int) -> str:
    h = int(hashlib.sha256(f"clarifybench_split_v1:{task_id}".encode()).hexdigest(), 16) % 100
    return "train" if h < train_pct else ("dev" if h < train_pct + dev_pct else "test")
