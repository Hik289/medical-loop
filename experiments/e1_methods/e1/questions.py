"""Structured question (HumanLoop §4.2) + Native / Choice rendering and response parsing."""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from .llm import LLMClient

QTYPES = ["scope", "constraint", "value", "temporal", "preference"]
OTHER_OPTIONS = [
    {"id": "insufficient", "text": "Insufficient Evidence"},
    {"id": "none", "text": "None of the Above"},
    {"id": "skip", "text": "Skip"},
]
RESPONSE_TYPES = ["substantive", "insufficient", "none", "skip", "free_text", "no_response"]


def make_question(text: str, origin: str, options: Optional[List[str]] = None, qtype: str = "value",
                  affected: str = "", evidence: Optional[List[str]] = None,
                  flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    opts = []
    for i, o in enumerate((options or [])[:3]):
        o = str(o).strip()
        if o and o not in [x["text"] for x in opts]:
            opts.append({"id": f"c{len(opts) + 1}", "text": o[:300], "local_value": o[:120]})
    qtype = qtype if qtype in QTYPES else "value"
    return {"question_type": qtype, "question": (text or "").strip()[:600], "evidence": evidence or [],
            "substantive_options": opts, "other_options": list(OTHER_OPTIONS),
            "affected_action_field": affected, "origin": origin, "flags": flags or {}}


def render_native(q: Dict[str, Any], show_options: bool) -> str:
    """Native protocol: free-text question as the method would ask it."""
    if not show_options or not q["substantive_options"]:
        return q["question"]
    opts = "; ".join(f"({o['id']}) {o['text']}" for o in q["substantive_options"])
    return f"{q['question']} Options: {opts}."


def render_choice(q: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Choice protocol: substantive options shuffled with stable ids, plus the 3 fixed options.
    No AI recommendation is shown (§4.2)."""
    subs = list(q["substantive_options"])
    random.Random(seed).shuffle(subs)
    lines = [q["question"]] + [f"[{o['id']}] {o['text']}" for o in subs] + \
            [f"[{o['id']}] {o['text']}" for o in q["other_options"]]
    return {"text": "\n".join(lines), "order": [o["id"] for o in subs]}


CHOICE_WRAP_SYS = (
    "You convert an assistant's clarification question into a local multiple-choice question. "
    "Write 2-3 mutually exclusive substantive options that each state ONE local judgement "
    "(a value, scope, constraint or interpretation) the user could pick. Do not write complete "
    "final answers and do not mark any option as recommended. Output JSON: "
    "{\"question\": str, \"options\": [str, str, (str)], \"question_type\": "
    "\"scope|constraint|value|temporal|preference\", \"affected_field\": str}"
)


def ensure_choice_options(llm: LLMClient, q: Dict[str, Any], context: str, purpose: str) -> Dict[str, Any]:
    """Choice adaptation for methods whose native question is free text (CLAM, AwN, native ask).
    The extra API call is counted under purpose 'method:choice_wrap'."""
    if len(q["substantive_options"]) >= 2:
        return q
    o = llm.json(CHOICE_WRAP_SYS, f"CONTEXT:\n{context}\n\nQUESTION: {q['question']}", purpose)
    opts = [str(x) for x in (o.get("options") or []) if str(x).strip()]
    if len(opts) < 2:
        return q  # caller records 'choice_wrap_failed'
    nq = make_question(o.get("question") or q["question"], q["origin"], opts,
                       o.get("question_type", q["question_type"]), o.get("affected_field", ""),
                       q.get("evidence"), q.get("flags"))
    nq["choice_wrapped"] = True
    return nq


def parse_choice(raw: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(raw.get("choice", "")).strip().lower().strip("[]() ")
    valid = {o["id"]: o["text"] for o in q["substantive_options"]}
    if cid in valid:
        return {"choice_id": cid, "r_type": "substantive", "text": valid[cid]}
    for o in q["other_options"]:
        if cid == o["id"] or cid == o["text"].lower():
            return {"choice_id": o["id"], "r_type": o["id"], "text": o["text"]}
    return {"choice_id": "skip", "r_type": "skip", "text": "Skip", "parse_failed": True, "raw": raw}


def choice_reply_text(parsed: Dict[str, Any]) -> str:
    """What the agent receives back after the human picks an option."""
    t = parsed["r_type"]
    if t == "substantive":
        return f"Selected option: {parsed['text']}"
    if t == "insufficient":
        return "The user indicated insufficient evidence to answer this question."
    if t == "none":
        return "The user indicated none of the offered options is correct."
    return "The user skipped this question."


def q_brief(q: Dict[str, Any]) -> str:
    return json.dumps({k: q[k] for k in ("question_type", "question", "substantive_options")},
                      ensure_ascii=False)
