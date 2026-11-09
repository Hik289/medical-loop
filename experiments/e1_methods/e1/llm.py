"""OpenAI-compatible chat client with per-role accounting.

* No hidden fallbacks: a call that still fails after retries raises LLMCallError and is
  recorded in the ledger; the worker turns that into task status "error".
* `purpose` is metadata only (never inserted into prompts); it lets the ledger split cost
  by role (agent / method / user_sim / judge) and lets fake responders script tests.
* Optional replay cache (prompt-hash keyed) is used ONLY for HumanChoice teacher branches
  so that branch replays share an identical prefix; cache hits are counted separately.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .config import api_key


class LLMCallError(RuntimeError):
    pass


def _role_of(purpose: str) -> str:
    return purpose.split(":", 1)[0] if ":" in purpose else purpose


class Ledger:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.by_role: Dict[str, Dict[str, float]] = {}
        self.errors: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []

    def add(self, purpose: str, ok: bool, usage: Dict[str, Any] | None, latency: float,
            cache_hit: bool = False, err: str | None = None) -> None:
        role = _role_of(purpose)
        with self.lock:
            r = self.by_role.setdefault(role, {"calls": 0, "live_calls": 0, "cache_hits": 0, "errors": 0,
                                               "prompt_tokens": 0, "completion_tokens": 0,
                                               "total_tokens": 0, "latency_s": 0.0})
            r["calls"] += 1
            if cache_hit:
                r["cache_hits"] += 1
            else:
                r["live_calls"] += 1
                r["latency_s"] += latency
            if not ok:
                r["errors"] += 1
                self.errors.append({"purpose": purpose, "error": err})
            if usage and not cache_hit:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    r[k] += int(usage.get(k) or 0)
            self.calls.append({"purpose": purpose, "ok": ok, "cache_hit": cache_hit,
                               "latency_s": round(latency, 3), "usage": usage})

    def summary(self, price_in: float | None = None, price_out: float | None = None) -> Dict[str, Any]:
        with self.lock:
            tot = {"calls": 0, "live_calls": 0, "cache_hits": 0, "errors": 0, "prompt_tokens": 0,
                   "completion_tokens": 0, "total_tokens": 0, "latency_s": 0.0}
            for r in self.by_role.values():
                for k in tot:
                    tot[k] += r[k]
            usd = None
            if price_in is not None and price_out is not None:
                usd = (tot["prompt_tokens"] * price_in + tot["completion_tokens"] * price_out) / 1e6
            return {"by_role": json.loads(json.dumps(self.by_role)), "total": tot, "usd": usd,
                    "n_errors": len(self.errors), "errors": self.errors[:20]}


class LLMClient:
    def __init__(self, cfg: Dict[str, Any], ledger: Ledger | None = None,
                 replay_cache_dir: str | None = None) -> None:
        self.cfg = cfg["llm"]
        self.ledger = ledger or Ledger()
        self.model = self.cfg["model"]
        self.replay_cache_dir = replay_cache_dir
        self._fake: Optional[Callable[[str, List[Dict[str, str]]], str]] = None
        prov = self.cfg.get("provider", "openai")
        if prov.startswith("fake:"):
            self._fake = _load_fake(prov.split(":", 1)[1])
            self._client = None
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.cfg["base_url"], api_key=api_key(cfg),
                                  max_retries=int(self.cfg.get("max_retries", 6)),
                                  timeout=float(self.cfg.get("request_timeout_s", 180)))

    # ------------------------------------------------------------------
    def meta(self) -> Dict[str, Any]:
        return {"model": self.model, "base_url": self.cfg.get("base_url"),
                "reasoning_effort": self.cfg.get("reasoning_effort"), "provider": self.cfg.get("provider")}

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.replay_cache_dir, key[:2], key + ".json")

    def chat(self, messages: List[Dict[str, str]], purpose: str) -> str:
        key = hashlib.sha256(json.dumps([self.model, messages], ensure_ascii=False,
                                        sort_keys=True).encode()).hexdigest()
        if self.replay_cache_dir:
            p = self._cache_path(key)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    content = json.load(f)["content"]
                self.ledger.add(purpose, True, None, 0.0, cache_hit=True)
                return content
        last_err = None
        for attempt in range(int(self.cfg.get("app_retries", 2)) + 1):
            t0 = time.time()
            try:
                if self._fake is not None:
                    content = self._fake(purpose, messages)
                    usage = {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                             "completion_tokens": len(content or "") // 4}
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                else:
                    kw: Dict[str, Any] = dict(model=self.model, messages=messages,
                                              timeout=float(self.cfg.get("request_timeout_s", 180)))
                    if self.cfg.get("reasoning_effort"):
                        kw["extra_body"] = {"reasoning_effort": self.cfg["reasoning_effort"]}
                    r = self._client.chat.completions.create(**kw)
                    content = r.choices[0].message.content
                    usage = r.usage.model_dump() if getattr(r, "usage", None) else None
                if content is None:
                    raise LLMCallError("empty content")
                self.ledger.add(purpose, True, usage, time.time() - t0)
                if self.replay_cache_dir:
                    p = self._cache_path(key)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    tmp = p + f".tmp{os.getpid()}"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump({"content": content, "purpose": purpose}, f, ensure_ascii=False)
                    os.replace(tmp, p)
                return content
            except Exception as e:  # noqa: BLE001 - every failure is recorded
                last_err = f"{type(e).__name__}: {str(e)[:300]}"
                self.ledger.add(purpose, False, None, time.time() - t0, err=last_err)
                time.sleep(min(2 * (attempt + 1), 6) if self._fake is None else 0)
        raise LLMCallError(f"{purpose}: {last_err}")

    def json(self, system: str, user: str, purpose: str) -> Dict[str, Any]:
        """Chat and parse the first JSON object. Parse failures return {} (not an API error)."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return parse_json(self.chat(msgs, purpose))


def parse_json(text: str | None) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {"_list": obj}
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _load_fake(path: str) -> Callable[[str, List[Dict[str, str]]], str]:
    spec = importlib.util.spec_from_file_location("e1_fake_responder", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.respond
