"""HumanChoice G/Q/A modules: fixed 64-d features, 64->64->32->1 MLPs (6,273 params each,
18,819 total), losses L_A / L_Q / L_G, and teacher-label construction from measured branches.

Pure numpy so the runner has no torch dependency. Normalisation statistics come from the
module's own training rows only (training-only normalization).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DIM = 64
N_PARAMS = 64 * 64 + 64 + 64 * 32 + 32 + 32 * 1 + 1  # 6273
QTYPES = ["scope", "constraint", "value", "temporal", "preference"]
FLAG_KEYS = ["changes_action", "info_only_user_knows", "derivable_by_tool", "evidence_missing",
             "options_conflict_with_proposed"]
RTYPES = ["substantive", "insufficient", "none", "skip", "free_text", "no_response"]
HILSQL_COMMANDS = {"execute_sql", "submit_sql", "get_table_info", "get_column_info", "get_business_info",
                   "get_database_info"}


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


# ------------------------------------------------------------------ features
def _args(a: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(a, dict):
        return {}
    if isinstance(a.get("arguments"), dict):
        return a["arguments"]
    return {"arg": a.get("arg")} if "arg" in a else {}


def _name(a: Optional[Dict[str, Any]]) -> str:
    if not isinstance(a, dict):
        return ""
    return str(a.get("tool_name") or a.get("command") or "")


def _is_final(a: Optional[Dict[str, Any]]) -> bool:
    return _name(a) in ("final_answer", "submit_sql")


def _in_text(v: Any, text: str) -> bool:
    s = str(v).strip().lower()
    return bool(s) and len(s) <= 200 and s in text.lower()


def state_features(state: Dict[str, Any]) -> List[float]:
    a0 = state.get("proposed") or {}
    args = _args(a0)
    req = state.get("request") or ""
    grounded = [1.0 if _in_text(v, req) else 0.0 for v in args.values()]
    return [
        1.0 if state.get("benchmark") == "clarifybench" else 0.0,
        1.0 if state.get("benchmark") == "hilsql" else 0.0,
        state.get("checkpoint_idx", 0) / 5.0,
        state.get("step_idx", 0) / 30.0,
        state.get("budget_left", 0) / 3.0,
        len(state.get("observations") or []) / 20.0,
        state.get("n_tool_errors", 0) / 5.0,
        min(len(req), 4000) / 2000.0,
        len(state.get("asked") or []) / 3.0,
        len(args) / 10.0,
        1.0 if _is_final(a0) else 0.0,
        state.get("request_idx", 0) / 3.0,
        min(len(json.dumps(a0, default=str)), 3000) / 1000.0,
        float(np.mean(grounded)) if grounded else 0.0,
        1.0,
        0.0,
    ]


def question_block(state: Dict[str, Any], q: Dict[str, Any]) -> List[float]:
    flags = q.get("flags") or {}
    req = state.get("request") or ""
    opts = q.get("substantive_options") or []
    args = _args(state.get("proposed"))
    aff = [x.strip() for x in str(q.get("affected_action_field", "")).split(",") if x.strip()]
    return ([1.0 if q.get("question_type") == t else 0.0 for t in QTYPES] + [
        min(len(q.get("question", "")), 600) / 300.0,
        len(opts) / 3.0,
        len(q.get("evidence") or []) / 5.0,
    ] + [1.0 if flags.get(k) else 0.0 for k in FLAG_KEYS] + [
        1.0 if any(a in args for a in aff) else 0.0,
        (sum(1 for o in opts if _in_text(o.get("text", ""), req)) / len(opts)) if opts else 0.0,
        state.get("pool_size", 0) / 3.0,
        state.get("pool_pos", 0) / 3.0,
    ])


def _pad(v: List[float]) -> List[float]:
    assert len(v) <= DIM, len(v)
    return [float(x) for x in v] + [0.0] * (DIM - len(v))


def q_features(state: Dict[str, Any], q: Dict[str, Any]) -> List[float]:
    return _pad(state_features(state) + question_block(state, q))


def g_features(state: Dict[str, Any], q: Dict[str, Any]) -> List[float]:
    # G conditions on (x_t, q_t): same deterministic block as Q, separate MLP.
    return q_features(state, q)


def a_features(state: Dict[str, Any], q: Dict[str, Any], reply: Dict[str, Any],
               a0: Dict[str, Any], a1: Optional[Dict[str, Any]]) -> List[float]:
    a1 = a1 or {}
    x0, x1 = _args(a0), _args(a1)
    keys = set(x0) | set(x1)
    diff = [k for k in keys if json.dumps(x0.get(k), default=str) != json.dumps(x1.get(k), default=str)]
    rtext = reply.get("text") or ""
    changed_vals = [x1[k] for k in diff if k in x1]
    return _pad(state_features(state) + question_block(state, q) + [
        1.0 if reply.get("r_type", "free_text") == t else 0.0 for t in RTYPES] + [
        1.0 if json.dumps(a0, sort_keys=True, default=str) == json.dumps(a1, sort_keys=True, default=str) else 0.0,
        1.0 if _name(a0) == _name(a1) else 0.0,
        len(diff) / max(1, len(keys)),
        len(x1) / 10.0,
        1.0 if _is_final(a1) else 0.0,
        1.0 if _is_final(a0) else 0.0,
        (sum(1 for v in changed_vals if _in_text(v, rtext)) / len(changed_vals)) if changed_vals else 0.0,
        min(len(rtext), 600) / 300.0,
        1.0 if action_is_valid(state, a1) else 0.0,
    ])


def action_is_valid(state: Dict[str, Any], a: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(a, dict) or not a:
        return False
    if state.get("benchmark") == "hilsql":
        return a.get("command") in HILSQL_COMMANDS and isinstance(a.get("arg", ""), str)
    names = set(state.get("tool_names") or []) | {"final_answer"}
    return a.get("tool_name") in names and isinstance(a.get("arguments", {}), dict)


# ------------------------------------------------------------------ MLP
class MLP:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)

        def he(i, o):
            return rng.normal(0, math.sqrt(2.0 / i), size=(i, o))
        self.p = {"W1": he(64, 64), "b1": np.zeros(64), "W2": he(64, 32), "b2": np.zeros(32),
                  "W3": he(32, 1), "b3": np.zeros(1)}
        self.mu = np.zeros(DIM)
        self.sd = np.ones(DIM)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    def _fwd(self, X: np.ndarray):
        h1 = np.maximum(0, X @ self.p["W1"] + self.p["b1"])
        h2 = np.maximum(0, h1 @ self.p["W2"] + self.p["b2"])
        return (h2 @ self.p["W3"] + self.p["b3"])[:, 0], (X, h1, h2)

    def _bwd(self, cache, dout: np.ndarray) -> Dict[str, np.ndarray]:
        X, h1, h2 = cache
        d3 = dout[:, None]
        g = {"W3": h2.T @ d3, "b3": d3.sum(0)}
        dh2 = (d3 @ self.p["W3"].T) * (h2 > 0)
        g["W2"], g["b2"] = h1.T @ dh2, dh2.sum(0)
        dh1 = (dh2 @ self.p["W2"].T) * (h1 > 0)
        g["W1"], g["b1"] = X.T @ dh1, dh1.sum(0)
        return g

    def norm(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def score(self, x: Sequence[float]) -> float:
        return float(self._fwd(self.norm(np.asarray([x], dtype=float)))[0][0])

    def to_dict(self) -> Dict[str, Any]:
        return {"params": {k: v.tolist() for k, v in self.p.items()}, "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MLP":
        m = cls()
        m.p = {k: np.asarray(v, dtype=float) for k, v in d["params"].items()}
        m.mu, m.sd = np.asarray(d["mu"], dtype=float), np.asarray(d["sd"], dtype=float)
        return m


class AdamW:
    def __init__(self, params, lr=1e-3, wd=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps, self.t = lr, wd, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * params[k])


def _fit_norm(m: MLP, X: np.ndarray) -> None:
    m.mu = X.mean(0)
    m.sd = X.std(0) + 1e-6


def train_binary(X: np.ndarray, y: np.ndarray, seed: int, epochs=30, lr=1e-3, bs=64) -> tuple[MLP, List[float]]:
    """L_A / L_G: Bernoulli NLL of the teacher target."""
    m = MLP(seed)
    _fit_norm(m, X)
    Xn = m.norm(X)
    opt = AdamW(m.p, lr)
    rng = np.random.default_rng(seed)
    curve = []
    for _ in range(epochs):
        idx = rng.permutation(len(Xn))
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            z, cache = m._fwd(Xn[b])
            p = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
            tot += float(-np.sum(y[b] * np.log(p + 1e-12) + (1 - y[b]) * np.log(1 - p + 1e-12)))
            opt.step(m.p, m._bwd(cache, (p - y[b]) / len(b)))
        curve.append(tot / len(Xn))
    return m, curve


def train_pairwise(Xp: np.ndarray, Xn_: np.ndarray, seed: int, epochs=30, lr=1e-3, bs=64) -> tuple[MLP, List[float]]:
    """L_Q: pairwise logistic loss -log sigma(s(q+) - s(q-)) (softmax over a 2-question pool)."""
    m = MLP(seed)
    _fit_norm(m, np.vstack([Xp, Xn_]))
    A, B = m.norm(Xp), m.norm(Xn_)
    opt = AdamW(m.p, lr)
    rng = np.random.default_rng(seed)
    curve = []
    for _ in range(epochs):
        idx = rng.permutation(len(A))
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            sp, cp = m._fwd(A[b])
            sn, cn = m._fwd(B[b])
            d = sp - sn
            s = 1 / (1 + np.exp(-np.clip(d, -60, 60)))
            tot += float(-np.sum(np.log(s + 1e-12)))
            gd = (s - 1) / len(b)
            gp, gn = m._bwd(cp, gd), m._bwd(cn, -gd)
            opt.step(m.p, {k: gp[k] + gn[k] for k in gp})
        curve.append(tot / len(A))
    return m, curve


# ------------------------------------------------------------------ module IO
def save_module(path: str, name: str, m: MLP, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"module": name, "n_params": m.n_params(), "meta": meta, **m.to_dict()}, f)


def load_modules(modules_dir: str, seed: int) -> Dict[str, MLP]:
    out = {}
    for name in ("G", "Q", "A"):
        p = os.path.join(modules_dir, f"{name}_s{seed}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing trained module {p}")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        m = MLP.from_dict(d)
        if m.n_params() != N_PARAMS:
            raise ValueError(f"{p}: {m.n_params()} params != {N_PARAMS}")
        out[name] = m
    return out


# ------------------------------------------------------------------ teacher labels (§8, §9)
def group_of(task_id: str) -> str:
    h = int(hashlib.sha256(f"hc_group:{task_id}".encode()).hexdigest(), 16)
    return ["D_A", "D_Q", "D_G"][h % 3]


def build_A(branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A target: for (task, cp, q) with both a0/a1 branches measured, a1 legal and Y differs."""
    rows, by = [], {}
    for b in branches:
        if b["branch"]["action"] == "ask" and b.get("a_features") is not None:
            by.setdefault((b["task_id"], b["branch"]["cp"], b["branch"]["q_index"]), {})[b["branch"]["a_choice"]] = b
    for (tid, cp, qi), d in by.items():
        if "a0" in d and "a1" in d and "a1" in (d["a1"].get("legal") or []) and d["a0"]["Y"] != d["a1"]["Y"]:
            rows.append({"task_id": tid, "x": d["a1"]["a_features"], "y": 1.0 if d["a1"]["Y"] > d["a0"]["Y"] else 0.0})
    return rows


def u_ask(by_q: Dict[str, Dict[str, Any]], A: Optional[MLP]) -> Optional[float]:
    """Measured U^Ask(q): Y of the branch whose action A would choose (a1 only if legal)."""
    a0, a1 = by_q.get("a0"), by_q.get("a1")
    if a0 is None:
        return None
    if a1 is None or "a1" not in (a1.get("legal") or []):
        return a0["Y"]
    choose_a1 = True if A is None else sigmoid(A.score(a1["a_features"])) >= 0.5
    return a1["Y"] if choose_a1 else a0["Y"]


def _by_cp(branches):
    out = {}
    for b in branches:
        key = (b["task_id"], b["branch"]["cp"])
        d = out.setdefault(key, {"wait": None, "ask": {}, "q_features": None})
        if b.get("q_features") is not None:
            d["q_features"] = b["q_features"]
        if b["branch"]["action"] == "wait":
            d["wait"] = b
        else:
            d["ask"].setdefault(b["branch"]["q_index"], {})[b["branch"]["a_choice"]] = b
    return out


def build_Q(branches, A: Optional[MLP]) -> List[Dict[str, Any]]:
    rows = []
    for (tid, cp), d in _by_cp(branches).items():
        qf = d["q_features"]
        if not qf:
            continue
        us = {qi: u_ask(v, A) for qi, v in d["ask"].items()}
        us = {k: v for k, v in us.items() if v is not None and k < len(qf)}
        for i in us:
            for j in us:
                if us[i] > us[j]:
                    rows.append({"task_id": tid, "x_pos": qf[i], "x_neg": qf[j]})
    return rows


def build_G(branches, A: Optional[MLP], Q: Optional[MLP]) -> List[Dict[str, Any]]:
    """G target uses the q_t that Q actually selects and its measured U^Ask (requirement (d))."""
    rows = []
    for (tid, cp), d in _by_cp(branches).items():
        qf = d["q_features"]
        if not qf or d["wait"] is None:
            continue
        qt = max(range(len(qf)), key=lambda i: Q.score(qf[i])) if Q is not None else 0
        if qt not in d["ask"]:
            continue
        ua = u_ask(d["ask"][qt], A)
        uw = d["wait"]["Y"]
        if ua is None or ua == uw:
            continue
        rows.append({"task_id": tid, "x": qf[qt], "y": 1.0 if ua > uw else 0.0})
    return rows
