"""Per-task JSON cache with atomic writes. Status is one of measured | error | timeout."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

STATUSES = ("measured", "error", "timeout")


def task_path(out_dir: str, benchmark: str, protocol: str, method: str, task_id: str,
              tag: str = "") -> str:
    m = method + (f"@{tag}" if tag else "")
    safe = task_id.replace("/", "__")
    return os.path.join(out_dir, benchmark, protocol, m, "tasks", f"{safe}.json")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def cached_status(path: str) -> Optional[str]:
    d = read_json(path)
    if not d:
        return None
    s = d.get("status")
    return s if s in STATUSES else None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def prompt_hash(*parts: str) -> str:
    return sha256_text("\n\x1e".join(parts))[:12]
