"""Benchmark-agnostic checkpoint controller shared by the ClarifyBench and HiL-SQL adapters.

Flow at a decision checkpoint (base agent B proposed a0):
    method.decide -> Wait (execute a0) | Ask(q) -> human reply r -> B proposes a1 from the
    same state + (q, r) -> A_t = legal subset of {a0, a1} -> chosen action.
HumanChoice chooses within A_t with its A module; every other method takes a1 (their
published behaviour: act on the clarified request).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import methods as M
from .questions import ensure_choice_options, make_question, render_choice, render_native


class Controller:
    def __init__(self, cfg: Dict[str, Any], method: str, protocol: str, benchmark: str, llm,
                 max_checkpoints: int, max_questions: int, task_seed: int,
                 forced: Optional[Dict[str, Any]] = None, question_scope: str = "request"):
        self.cfg, self.method_name, self.protocol, self.benchmark = cfg, method, protocol, benchmark
        self.llm = llm
        self.method = M.build(method, cfg)
        self.max_cp, self.max_q = max_checkpoints, max_questions
        self.task_seed = task_seed
        self.forced = forced
        self.scope = question_scope
        self.request_idx = -1
        self.cp = 0
        self.q_used_request = 0
        self.q_used_task = 0
        self.pending: Optional[Dict[str, Any]] = None
        self.asked: List[Dict[str, Any]] = []
        self.trace: List[Dict[str, Any]] = []
        self.counters = {"checkpoints": 0, "questions": 0, "choice_wrap_failed": 0, "action_overrides": 0,
                         "a1_adopted": 0, "a0_kept": 0, "base_ask_suppressed": 0}
        self.teacher: Dict[str, Any] = {}

    # ------------------------------------------------------------ bookkeeping
    def new_request(self) -> None:
        self.request_idx += 1
        self.cp = 0
        self.q_used_request = 0
        self.pending = None

    def native_may_ask(self) -> bool:
        """native_ask_human obeys the same question budget unless budget.enforce_on_native is false."""
        return self.budget_left() > 0 or not self.cfg["budget"].get("enforce_on_native", True)

    def budget_left(self) -> int:
        used = self.q_used_request if self.scope == "request" else self.q_used_task
        return max(0, self.max_q - used)

    def _context_text(self, state: Dict[str, Any]) -> str:
        return M.ctx(state, max_obs=6)

    def _present(self, q: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.protocol == "choice":
            q = ensure_choice_options(self.llm, q, self._context_text(state), "method:choice_wrap")
            if len(q["substantive_options"]) < 2:
                self.counters["choice_wrap_failed"] += 1
                return None
            rc = render_choice(q, seed=self.task_seed * 1000 + len(self.asked))
            return {"q": q, "rendered": rc["text"], "order": rc["order"]}
        show = self.method_name == "humanchoice"  # HumanChoice's question is MC by construction
        return {"q": q, "rendered": render_native(q, show), "order": None}

    # ------------------------------------------------------------ main hook
    def on_proposal(self, state: Dict[str, Any], is_checkpoint: bool) -> Dict[str, Any]:
        state = dict(state, benchmark=self.benchmark, request_idx=self.request_idx, asked=list(self.asked),
                     budget_left=self.budget_left())
        if self.pending is not None and self.pending.get("reply") is not None:
            return self._resolve(state)
        self.pending = None
        if not is_checkpoint or self.cp >= self.max_cp:
            return {"type": "proceed", "action": state["proposed"]}
        cp = self.cp
        self.cp += 1
        self.counters["checkpoints"] += 1
        state["checkpoint_idx"] = cp
        rec = {"request_idx": self.request_idx, "cp": cp, "step_idx": state.get("step_idx"),
               "a0": state["proposed"]}
        can_ask = self.budget_left() > 0
        forced = self._forced_for(cp)
        if forced == "pre_wait":
            rec.update(decision="wait", forced="pre_wait")
            self.trace.append(rec)
            return {"type": "proceed", "action": state["proposed"]}
        if not can_ask and self.method_name != "compute_matched":
            rec.update(decision="wait", reason="no_question_budget")
            self.trace.append(rec)
            return {"type": "proceed", "action": state["proposed"]}
        if forced is not None:
            dec = self.method.decide(self.llm, state, forced=forced)
            self.teacher.update(q_features=dec["signals"].get("q_features"), branch=forced,
                                pool_size=dec["signals"].get("pool_size", 0),
                                branch_invalid=bool(dec["signals"].get("branch_invalid")))
        else:
            dec = self.method.decide(self.llm, state)
        rec["signals"] = _slim(dec.get("signals"))
        if dec.get("action") is not None:
            self.counters["action_overrides"] += 1
            rec.update(decision="override", action=dec["action"])
            self.trace.append(rec)
            return {"type": "proceed", "action": dec["action"]}
        if not dec.get("ask") or not can_ask:
            rec["decision"] = "wait"
            self.trace.append(rec)
            return {"type": "proceed", "action": state["proposed"]}
        pres = self._present(dec["question"], state)
        if pres is None:
            rec["decision"] = "wait_choice_wrap_failed"
            self.trace.append(rec)
            return {"type": "proceed", "action": state["proposed"]}
        self.q_used_request += 1
        self.q_used_task += 1
        self.counters["questions"] += 1
        self.pending = {"a0": state["proposed"], "q": pres["q"], "rendered": pres["rendered"],
                        "q_state": dec.get("q_state", state), "cp": cp, "reply": None}
        rec.update(decision="ask", question=pres["q"], rendered=pres["rendered"])
        self.trace.append(rec)
        return {"type": "ask", "text": pres["rendered"], "question": pres["q"]}

    def native_question(self, text: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """native_ask_human: the base agent asked by itself; register it (choice-wrapped if needed)."""
        q = make_question(text, "native_ask_human")
        pres = self._present(q, dict(state, benchmark=self.benchmark, asked=list(self.asked),
                                     proposed=state.get("proposed") or {}))
        if pres is None:
            return None
        self.counters["questions"] += 1
        self.q_used_request += 1
        self.q_used_task += 1
        self.pending = {"a0": None, "q": pres["q"], "rendered": pres["rendered"], "native": True,
                        "cp": None, "reply": None, "original_text": text}
        self.trace.append({"request_idx": self.request_idx, "decision": "native_ask", "question": pres["q"],
                           "rendered": pres["rendered"]})
        return self.pending

    def on_reply(self, reply: Dict[str, Any]) -> None:
        if self.pending is None:
            return
        self.pending["reply"] = reply
        self.asked.append({"question": self.pending["q"]["question"], "reply": reply.get("text", ""),
                           "affected": self.pending["q"].get("affected_action_field", ""),
                           "r_type": reply.get("r_type")})
        self.trace.append({"request_idx": self.request_idx, "event": "reply", "reply": reply})
        if self.pending.get("native"):
            self.pending = None

    def _resolve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        p, self.pending = self.pending, None
        a0, a1, reply = p["a0"], state["proposed"], p["reply"]
        if self.method_name == "humanchoice":
            from . import hc
            qs = dict(p["q_state"], benchmark=self.benchmark)
            forced_a = self.forced.get("a_choice") if self.forced and p["cp"] == self.forced.get("cp") \
                and self.request_idx == self.forced.get("request_idx", 0) else None
            ch = self.method.choose_action(qs, p["q"], reply, a0, a1, forced=forced_a)
            if forced_a is not None:
                self.teacher.update(a_features=hc.a_features(qs, p["q"], reply, a0, a1), legal=ch["legal"],
                                    a_choice=forced_a)
            chosen = a1 if ch["choice"] == "a1" else a0
        else:
            ch, chosen = {"choice": "a1", "legal": ["a0", "a1"]}, a1
        self.counters["a1_adopted" if ch["choice"] == "a1" else "a0_kept"] += 1
        self.trace.append({"request_idx": self.request_idx, "event": "action_select", "a0": a0, "a1": a1,
                           "choice": ch})
        return {"type": "proceed", "action": chosen}

    def _forced_for(self, cp: int):
        if not self.forced or self.request_idx != self.forced.get("request_idx", 0):
            return None
        if cp < self.forced["cp"]:
            return "pre_wait"
        if cp == self.forced["cp"]:
            return self.forced
        return None

    def summary(self) -> Dict[str, Any]:
        return {"counters": dict(self.counters), "asked": self.asked,
                "adaptation": M.ADAPTATION_NOTES.get(self.method_name)}


def _slim(sig: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not sig:
        return {}
    out = {k: v for k, v in sig.items() if k != "q_features"}
    s = json.dumps(out, default=str)
    return out if len(s) < 20000 else {"truncated": s[:20000]}
