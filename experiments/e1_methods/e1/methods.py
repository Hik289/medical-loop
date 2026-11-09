"""Ask-method decision policies (benchmark-agnostic).

Each policy sees a DecisionState at a decision checkpoint (the base agent B has proposed
an action a0) and returns a Decision:
    {"ask": bool, "question": q|None, "action": a0' | None, "signals": {...}}
All signals come from real API outputs on that state; the numbers derived from them
(EIG, EVPI, VoI, rewards) are deterministic functions documented next to each method.

API adaptations (gpt-5.6-sol exposes no token logprobs and rejects temperature/n):
  * probabilities are verbalized by the model in structured output;
  * "sampling" of candidate solutions/hypotheses is one call that lists N distinct
    candidates (the deterministic endpoint returns identical samples otherwise).
These are labelled in every task record via ADAPTATION_NOTES.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .questions import make_question

ADAPTATION_NOTES = {
    "autonomous": "Same base agent with the human channel removed (ask_user virtual tool disabled; "
                  "tool errors are fed back as observations instead of triggering a question).",
    "native_ask_human": "Official base agent unchanged; it may call ask_user / ask_human whenever it wants.",
    "compute_matched": "Autonomous branch-and-aggregate: K candidate actions listed in one call + one "
                       "aggregation call per checkpoint and round; no human. Match K/rounds to the "
                       "measured extra calls of the ask methods (README).",
    "clam_api": "CLAM-API: ambiguity score is a verbalized probability (no token logprob); ask once per "
                "request at its first checkpoint iff p_ambiguous > tau; clarifying question generated "
                "by the same API.",
    "awn": "Ask-when-Needed: before every tool call the API checks each argument for missing / "
           "incorrect values w.r.t. the user's instruction and asks if any is found.",
    "uot": "UoT: hypothesis set = N candidate actions; M yes/no questions; hypothesis partitions by the "
           "API; uniform-prior information gain with depth-2 reward propagation; ask argmax iff reward>min.",
    "atd": "ATD: N candidate solutions (listed in one call); M questions; per-candidate implied answer "
           "by the API; EIG = entropy of the induced answer partition; ask argmax iff EIG>min.",
    "sage": "SAGE-Agent (inference-time, no GRPO): verbalized per-argument value distributions; "
            "P(correct)=prod max_v p(v); EVPI(S)=prod_{i not in S} max p_i - prod_i max p_i; "
            "cost=alpha*|S| + redundancy penalty for re-asked args; ask iff best EVPI-cost>tau.",
    "voi": "VoI: belief = N hypotheses with verbalized probabilities; deterministic reply model from the "
           "API; U(no ask)=max_h p(h); U(ask q)=sum_r P(r) max_h p(h|r); ask iff U(ask)-U(no ask)-c>0.",
    "humanchoice": "HumanChoice: API proposes <=3 local MC questions with schema flags; trained Q (softmax), "
                   "G (Ask/Wait) and A (a0 vs a1) MLPs (64-64-32-1) decide; no heuristic fallback.",
}

ACTION_FORMAT = {
    "clarifybench": '{"tool_name": str, "arguments": {...}}',
    "hilsql": '{"command": "execute_sql|submit_sql|get_table_info|get_column_info|get_business_info|'
              'get_database_info", "arg": str}',
}

QUESTION_RULES = (
    "Questions go to the user/stakeholder who knows the intended task details. A question must be "
    "concise, self-contained, about ONE specific point, and use what/which/how/when (no embedded "
    "assumptions, no request for best practices)."
)


def ctx(state: Dict[str, Any], max_obs: int = 12) -> str:
    obs = state.get("observations") or []
    obs_txt = "\n".join(f"- {o[:600]}" for o in obs[-max_obs:]) or "None"
    asked = state.get("asked") or []
    asked_txt = "\n".join(f"- Q: {a['question']} | A: {a['reply']}" for a in asked) or "None"
    return (f"USER REQUEST:\n{state['request']}\n\nAVAILABLE TOOLS:\n{state['tools_text']}\n\n"
            f"OBSERVATIONS SO FAR:\n{obs_txt}\n\nQUESTIONS ALREADY ASKED:\n{asked_txt}\n\n"
            f"PROPOSED NEXT ACTION:\n{json.dumps(state['proposed'], ensure_ascii=False)}")


def akey(a: Any) -> str:
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


def norm_ans(s: Any) -> str:
    return " ".join(str(s).strip().lower().rstrip(".").split())


def entropy(counts: List[float]) -> float:
    tot = float(sum(counts))
    return -sum((c / tot) * math.log(c / tot) for c in counts if c > 0) if tot > 0 else 0.0


def wait(signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ask": False, "question": None, "action": None, "signals": signals or {}}


# ---------------------------------------------------------------- shared generators
CAND_SYS = (
    "You enumerate distinct plausible interpretations of what the user wants at this step. List up to "
    "{n} DISTINCT complete candidate next actions in the format {fmt}, each with your probability that it "
    "is what the user intends (probabilities sum to 1; most likely first). Candidates must differ in "
    "tool/command or argument values. Output JSON: {{\"candidates\": [{{\"action\": ..., \"prob\": float}}]}}"
)


def gen_candidates(llm: LLMClient, state: Dict[str, Any], n: int, purpose: str) -> List[Dict[str, Any]]:
    o = llm.json(CAND_SYS.format(n=n, fmt=ACTION_FORMAT[state["benchmark"]]), ctx(state), purpose)
    out, seen = [], set()
    for c in (o.get("candidates") or [])[:n]:
        if not isinstance(c, dict) or not isinstance(c.get("action"), dict):
            continue
        k = akey(c["action"])
        if k in seen:
            continue
        seen.add(k)
        try:
            p = max(0.0, float(c.get("prob", 0)))
        except (TypeError, ValueError):
            p = 0.0
        out.append({"action": c["action"], "prob": p})
    z = sum(c["prob"] for c in out)
    for c in out:
        c["prob"] = c["prob"] / z if z > 0 else 1.0 / len(out)
    return out


def _cands_txt(cands: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{i}] {akey(c['action'])}" for i, c in enumerate(cands))


# ---------------------------------------------------------------- methods
class Method:
    name = "base"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.p = cfg["methods"].get(self.name, {})

    def decide(self, llm: LLMClient, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class Autonomous(Method):
    name = "autonomous"

    def decide(self, llm, state):
        return wait()


class NativeAskHuman(Autonomous):
    """Asking is done by the base agent itself (official ask tool); no policy decisions."""
    name = "native_ask_human"


class ComputeMatched(Method):
    name = "compute_matched"
    AGG_SYS = ("You are given several candidate next actions for the same step. Compare them against the "
               "user's request, tool documentation and observations, then output the single best action "
               "(you may merge argument values when candidates complement each other). Output JSON: "
               "{\"action\": " + "<action>" + ", \"reason\": str}")

    def decide(self, llm, state):
        best = state["proposed"]
        rounds = []
        for r in range(int(self.p.get("rounds", 1))):
            st = dict(state, proposed=best)
            cands = gen_candidates(llm, st, int(self.p.get("k", 4)), f"method:cm_candidates_r{r}")
            pool = [{"action": best, "prob": None}] + [c for c in cands if akey(c["action"]) != akey(best)]
            o = llm.json(self.AGG_SYS, ctx(st) + "\n\nCANDIDATES:\n" + _cands_txt(pool), f"method:cm_aggregate_r{r}")
            if isinstance(o.get("action"), dict) and o["action"]:
                best = o["action"]
            rounds.append({"n_candidates": len(pool), "chosen": best})
        changed = akey(best) != akey(state["proposed"])
        return {"ask": False, "question": None, "action": best if changed else None,
                "signals": {"rounds": rounds, "changed": changed}}


class ClamAPI(Method):
    name = "clam_api"
    DET_SYS = ("Decide whether the user's request is ambiguous for completing the current step, i.e. whether "
               "different reasonable readings would lead to different tool calls / answers. Output JSON: "
               "{\"p_ambiguous\": float in [0,1], \"ambiguity\": str}")
    Q_SYS = ("The user's request is ambiguous. Ask ONE clarifying question that resolves the ambiguity "
             "you identified. " + QUESTION_RULES + " Output JSON: {\"question\": str}")

    def decide(self, llm, state):
        if state["checkpoint_idx"] != 0:  # CLAM clarifies the incoming request once
            return wait({"skipped": "clam asks only at the first checkpoint of a request"})
        o = llm.json(self.DET_SYS, ctx(state), "method:clam_detect")
        try:
            p = float(o.get("p_ambiguous"))
        except (TypeError, ValueError):
            return wait({"parse_failed": True, "raw": o})
        sig = {"p_ambiguous": p, "tau": self.p.get("tau", 0.5), "ambiguity": o.get("ambiguity")}
        if p <= float(self.p.get("tau", 0.5)):
            return wait(sig)
        q = llm.json(self.Q_SYS, ctx(state) + f"\n\nAMBIGUITY: {o.get('ambiguity')}", "method:clam_question")
        if not q.get("question"):
            return wait(dict(sig, question_parse_failed=True))
        return {"ask": True, "question": make_question(q["question"], self.name), "action": None, "signals": sig}


class AwN(Method):
    name = "awn"
    SYS = ("Before executing the proposed tool call, check every argument: is its value explicitly provided "
           "by the user, given in the observations, or a documented default that is acceptable? Flag "
           "arguments that are MISSING (user never specified and no acceptable default) or INCORRECT "
           "(contradicts the user's instruction or tool documentation). If any is flagged, write ONE "
           "question asking the user for exactly that information. " + QUESTION_RULES + " Output JSON: "
           "{\"missing\": [arg], \"incorrect\": [arg], \"needs_clarification\": bool, \"question\": str}")

    def decide(self, llm, state):
        o = llm.json(self.SYS, ctx(state), "method:awn_check")
        sig = {"missing": o.get("missing") or [], "incorrect": o.get("incorrect") or [],
               "needs_clarification": bool(o.get("needs_clarification"))}
        if not sig["needs_clarification"] or not o.get("question"):
            return wait(sig)
        args = list(sig["missing"]) + list(sig["incorrect"])
        return {"ask": True, "question": make_question(o["question"], self.name, affected=",".join(map(str, args))),
                "action": None, "signals": sig}


class UoT(Method):
    name = "uot"
    Q_SYS = ("Hypotheses about the intended next action are listed. Propose {m} YES/NO questions to the user "
             "whose answers would split the hypotheses as evenly as possible. For every question list the "
             "indices of hypotheses for which the user's answer would be YES. " + QUESTION_RULES +
             " Output JSON: {{\"questions\": [{{\"question\": str, \"yes\": [int]}}]}}")

    @staticmethod
    def r1(yes: frozenset, S: frozenset) -> float:
        if len(S) <= 1:
            return 0.0
        a, b = len(S & yes), len(S - yes)
        return math.log(len(S)) - sum((k / len(S)) * math.log(k) for k in (a, b) if k > 0)

    @classmethod
    def rewards(cls, parts: List[frozenset], n: int) -> List[float]:
        S = frozenset(range(n))
        out = []
        for i, y in enumerate(parts):
            r = cls.r1(y, S)
            for sub in (S & y, S - y):
                if len(sub) > 1:
                    nxt = max([cls.r1(y2, sub) for j, y2 in enumerate(parts) if j != i] or [0.0])
                    r += (len(sub) / len(S)) * nxt
            out.append(r)
        return out

    def decide(self, llm, state):
        hyps = gen_candidates(llm, state, int(self.p.get("n_hypotheses", 5)), "method:uot_hypotheses")
        if len(hyps) < 2:
            return wait({"n_hypotheses": len(hyps)})
        o = llm.json(self.Q_SYS.format(m=int(self.p.get("n_questions", 4))),
                     ctx(state) + "\n\nHYPOTHESES:\n" + _cands_txt(hyps), "method:uot_questions")
        qs = [q for q in (o.get("questions") or []) if isinstance(q, dict) and q.get("question")]
        if not qs:
            return wait({"n_hypotheses": len(hyps), "questions_parse_failed": True})
        parts = [frozenset(int(i) for i in (q.get("yes") or []) if str(i).lstrip("-").isdigit()
                           and 0 <= int(i) < len(hyps)) for q in qs]
        rw = self.rewards(parts, len(hyps))
        imm = [self.r1(y, frozenset(range(len(hyps)))) for y in parts]
        j = max(range(len(qs)), key=lambda i: (rw[i], imm[i]))  # tie-break on immediate gain
        sig = {"n_hypotheses": len(hyps), "rewards": rw, "immediate_ig": imm, "chosen": j}
        if rw[j] <= float(self.p.get("min_reward", 1e-6)) or imm[j] <= 0:
            return wait(sig)
        return {"ask": True, "question": make_question(qs[j]["question"], self.name, ["Yes", "No"]),
                "action": None, "signals": sig}


class ATD(Method):
    name = "atd"
    Q_SYS = ("Candidate solutions for the next action are listed. Propose {m} questions to the user that "
             "would best discriminate between the candidates. " + QUESTION_RULES +
             " Output JSON: {{\"questions\": [str]}}")
    P_SYS = ("For each question and each candidate solution, state the short answer the user would give IF "
             "that candidate were exactly what they intend. Use identical wording for identical answers. "
             "Output JSON: {\"answers\": [{\"question_idx\": int, \"per_candidate\": [str]}]}")

    def decide(self, llm, state):
        cands = gen_candidates(llm, state, int(self.p.get("n_candidates", 5)), "method:atd_candidates")
        if len(cands) < 2:
            return wait({"n_candidates": len(cands), "eig": 0.0})
        o = llm.json(self.Q_SYS.format(m=int(self.p.get("n_questions", 4))),
                     ctx(state) + "\n\nCANDIDATES:\n" + _cands_txt(cands), "method:atd_questions")
        qs = [str(q) for q in (o.get("questions") or []) if str(q).strip()]
        if not qs:
            return wait({"n_candidates": len(cands), "questions_parse_failed": True})
        qtxt = "\n".join(f"Q{i}: {q}" for i, q in enumerate(qs))
        pa = llm.json(self.P_SYS, "CANDIDATES:\n" + _cands_txt(cands) + "\n\nQUESTIONS:\n" + qtxt,
                      "method:atd_partition")
        eig, groups = [0.0] * len(qs), [[] for _ in qs]
        for row in pa.get("answers") or []:
            try:
                i = int(row.get("question_idx"))
            except (TypeError, ValueError):
                continue
            ans = [norm_ans(a) for a in (row.get("per_candidate") or [])][:len(cands)]
            if 0 <= i < len(qs) and len(ans) == len(cands):
                cnt = Counter(ans)
                eig[i] = entropy(list(cnt.values()))  # uniform weights over sampled solutions
                groups[i] = [a for a, _ in cnt.most_common(3)]
        j = max(range(len(qs)), key=lambda i: eig[i])
        sig = {"n_candidates": len(cands), "eig": eig, "chosen": j}
        if eig[j] <= float(self.p.get("min_eig", 1e-6)):
            return wait(sig)
        return {"ask": True, "question": make_question(qs[j], self.name, groups[j]), "action": None, "signals": sig}


class SAGE(Method):
    name = "sage"
    U_SYS = ("Quantify structured uncertainty of the proposed action. For EVERY argument give up to 3 "
             "candidate values with probabilities (sum to 1) that it is the value the user intends, and "
             "whether the chosen value is explicitly grounded in the request/observations. Output JSON: "
             "{\"arguments\": [{\"name\": str, \"candidates\": [{\"value\": any, \"prob\": float}], "
             "\"grounded\": bool}]}")
    Q_SYS = ("Ask the user ONE question that resolves the value(s) of these arguments: {args}. Offer the "
             "listed candidate values as the answer options. " + QUESTION_RULES +
             " Output JSON: {{\"question\": str, \"options\": [str]}}")

    @staticmethod
    def evpi_table(maxp: Dict[str, float], max_subset: int) -> List[Dict[str, Any]]:
        names = sorted(maxp)
        base = math.prod(maxp.values()) if maxp else 1.0
        rows = []
        for k in range(1, max_subset + 1):
            for S in combinations(names, k):
                rest = math.prod(maxp[n] for n in names if n not in S)
                rows.append({"subset": list(S), "evpi": rest - base})
        return rows

    def decide(self, llm, state):
        o = llm.json(self.U_SYS, ctx(state), "method:sage_uncertainty")
        maxp, cands = {}, {}
        for a in o.get("arguments") or []:
            if not isinstance(a, dict) or not a.get("name"):
                continue
            ps = []
            for c in a.get("candidates") or []:
                try:
                    ps.append((float(c.get("prob", 0)), c.get("value")))
                except (TypeError, ValueError, AttributeError):
                    pass
            z = sum(p for p, _ in ps)
            if z <= 0:
                continue
            ps.sort(key=lambda t: -t[0])
            maxp[str(a["name"])] = ps[0][0] / z
            cands[str(a["name"])] = [v for _, v in ps]
        if not maxp:
            return wait({"parse_failed": True})
        asked_args = set()
        for a in state.get("asked") or []:
            asked_args.update(a.get("affected", "").split(","))
        rows = self.evpi_table(maxp, int(self.p.get("max_subset", 2)))
        for r in rows:
            red = float(self.p.get("redundancy_penalty", 1.0)) if set(r["subset"]) & asked_args else 0.0
            r["score"] = r["evpi"] - float(self.p.get("alpha_per_arg", 0.02)) * len(r["subset"]) - red
        best = max(rows, key=lambda r: r["score"])
        sig = {"p_correct": math.prod(maxp.values()), "max_prob": maxp, "best": best}
        if best["score"] <= float(self.p.get("tau", 0.1)):
            return wait(sig)
        args = best["subset"]
        opts_hint = {a: cands[a] for a in args}
        q = llm.json(self.Q_SYS.format(args=json.dumps(opts_hint, ensure_ascii=False, default=str)),
                     ctx(state), "method:sage_question")
        if not q.get("question"):
            return wait(dict(sig, question_parse_failed=True))
        opts = [str(x) for x in (q.get("options") or [])]
        if len(opts) < 2 and len(args) == 1:
            opts = [str(v) for v in cands[args[0]]]
        return {"ask": True, "question": make_question(q["question"], self.name, opts, "value", ",".join(args)),
                "action": None, "signals": sig}


class VoI(Method):
    name = "voi"
    R_SYS = ("Belief hypotheses about the intended next action are listed. Propose {m} questions to the user. "
             "For each question give the reply the user would give under EACH hypothesis (identical wording "
             "for identical replies). " + QUESTION_RULES +
             " Output JSON: {{\"questions\": [{{\"question\": str, \"replies\": [str]}}]}}")

    @staticmethod
    def value(probs: List[float], replies: List[str]) -> float:
        groups: Dict[str, List[float]] = {}
        for p, r in zip(probs, replies):
            groups.setdefault(r, []).append(p)
        return sum(max(g) for g in groups.values()) - max(probs)  # sum_r P(r) max_h p(h|r) - max_h p(h)

    def decide(self, llm, state):
        hyps = gen_candidates(llm, state, int(self.p.get("n_hypotheses", 5)), "method:voi_belief")
        if len(hyps) < 2:
            return wait({"n_hypotheses": len(hyps)})
        o = llm.json(self.R_SYS.format(m=int(self.p.get("n_questions", 3))),
                     ctx(state) + "\n\nHYPOTHESES (with belief):\n" +
                     "\n".join(f"[{i}] p={h['prob']:.3f} {akey(h['action'])}" for i, h in enumerate(hyps)),
                     "method:voi_replies")
        probs = [h["prob"] for h in hyps]
        rows = []
        for q in o.get("questions") or []:
            if not isinstance(q, dict) or not q.get("question"):
                continue
            reps = [norm_ans(r) for r in (q.get("replies") or [])]
            if len(reps) != len(hyps):
                continue
            rows.append({"question": q["question"], "voi": self.value(probs, reps) - float(self.p.get("comm_cost", 0.05)),
                         "replies": [r for r, _ in Counter(reps).most_common(3)]})
        if not rows:
            return wait({"n_hypotheses": len(hyps), "questions_parse_failed": True})
        best = max(rows, key=lambda r: r["voi"])
        sig = {"belief": probs, "net_voi": [r["voi"] for r in rows], "comm_cost": self.p.get("comm_cost", 0.05)}
        if best["voi"] <= 0:
            return wait(sig)
        return {"ask": True, "question": make_question(best["question"], self.name, best["replies"]),
                "action": None, "signals": sig}


class HumanChoice(Method):
    name = "humanchoice"
    POOL_SYS = ("Propose up to {k} candidate LOCAL multiple-choice questions for the user about judgements that "
                "affect the proposed action (scope, constraint, value, temporal, preference). Each has 2-3 "
                "substantive options stating one local judgement each (never full final answers). Also fill "
                "the schema flags honestly. " + QUESTION_RULES + " Output JSON: {{\"questions\": [{{"
                "\"question_type\": \"scope|constraint|value|temporal|preference\", \"question\": str, "
                "\"options\": [str], \"affected_action_field\": str, \"evidence\": [str], \"flags\": {{"
                "\"changes_action\": bool, \"info_only_user_knows\": bool, \"derivable_by_tool\": bool, "
                "\"evidence_missing\": bool, \"options_conflict_with_proposed\": bool}}}}]}}")

    def __init__(self, cfg):
        super().__init__(cfg)
        from . import hc
        self.hc = hc
        self.policy = self.p.get("policy", "trained")
        self.modules = None
        if self.policy == "trained":
            if not self.p.get("modules_dir"):
                raise RuntimeError("humanchoice requires methods.humanchoice.modules_dir (trained G/Q/A); "
                                   "no heuristic fallback is allowed")
            self.modules = hc.load_modules(self.p["modules_dir"], int(self.p.get("seed", 42)))

    def pool(self, llm, state) -> List[Dict[str, Any]]:
        o = llm.json(self.POOL_SYS.format(k=int(self.p.get("max_pool", 3))), ctx(state), "method:hc_pool")
        out = []
        for q in (o.get("questions") or [])[: int(self.p.get("max_pool", 3))]:
            if isinstance(q, dict) and q.get("question") and len(q.get("options") or []) >= 2:
                out.append(make_question(q["question"], self.name, [str(x) for x in q["options"]],
                                         q.get("question_type", "value"), str(q.get("affected_action_field", "")),
                                         [str(e) for e in (q.get("evidence") or [])], q.get("flags") or {}))
        return out

    def decide(self, llm, state, forced: Optional[Dict[str, Any]] = None):
        qs = self.pool(llm, state)
        if not qs:
            return wait({"pool_size": 0})
        sts = [dict(state, pool_size=len(qs), pool_pos=i) for i in range(len(qs))]
        fq = [self.hc.q_features(st, q) for st, q in zip(sts, qs)]
        sig: Dict[str, Any] = {"pool": qs, "pool_size": len(qs), "q_features": fq}
        if forced is not None:  # teacher branch (training split only)
            j = int(forced.get("q_index", 0))
            if forced["action"] == "wait" or j >= len(qs):
                return wait(dict(sig, forced=forced, branch_invalid=j >= len(qs)))
            sig.update(forced=forced, q_index=j)
            return {"ask": True, "question": qs[j], "action": None, "signals": sig, "q_state": sts[j]}
        if self.policy == "init":  # pi_0 used only for EI iteration-0 collection on train split
            j, p_ask = 0, 1.0
        else:
            qprobs = self.hc.softmax([self.modules["Q"].score(x) for x in fq])
            j = max(range(len(qs)), key=lambda i: qprobs[i])
            p_ask = self.hc.sigmoid(self.modules["G"].score(self.hc.g_features(sts[j], qs[j])))
            sig["q_probs"] = qprobs
        sig.update(q_index=j, p_ask=p_ask)
        if p_ask < 0.5:
            return wait(sig)
        return {"ask": True, "question": qs[j], "action": None, "signals": sig, "q_state": sts[j]}

    def choose_action(self, state, q, reply, a0, a1, forced: Optional[str] = None) -> Dict[str, Any]:
        """A module: pick within the legal set A_t subset of {a0, a1} (§7.3)."""
        legal = ["a0"]
        rt = reply.get("r_type", "free_text")
        if a1 is not None and rt not in ("skip", "no_response", "insufficient") and \
                self.hc.action_is_valid(state, a1):
            legal.append("a1")
        if forced in legal:
            return {"choice": forced, "legal": legal, "forced": True}
        if len(legal) == 1:
            return {"choice": "a0", "legal": legal}
        if self.policy == "init":
            return {"choice": "a1", "legal": legal, "p_a1": 1.0}
        p = self.hc.sigmoid(self.modules["A"].score(self.hc.a_features(state, q, reply, a0, a1)))
        return {"choice": "a1" if p >= 0.5 else "a0", "legal": legal, "p_a1": p}


REGISTRY = {c.name: c for c in [Autonomous, NativeAskHuman, ComputeMatched, ClamAPI, AwN, UoT, ATD, SAGE, VoI,
                                HumanChoice]}


def build(name: str, cfg: Dict[str, Any]) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name}")
    return REGISTRY[name](cfg)
