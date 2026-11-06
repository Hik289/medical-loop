#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autonomous_agent.py — H0.anchor_1 Autonomous Agent 复现 (命门, debug_move=COMPARE)

固定 gpt-5.6-sol 的完全自主 Agent, 在 270 test 上做三分类 (A/B/C), 仅用 34(实为32) 可见字段。
- 禁止泄漏: 只喂 agent_visible 字段 (剔除 16 outcome + 4 answer + 2 ID)
- Structured JSON 输出: {label: A|B|C, evidence_ids, uncertainty}
- 评估: overall acc + macro-F1 + per-class recall (AKI=C 单列) via eval_metrics
- reference: clean 监督锚 LogReg 71.5% acc / AKI recall 40.5%
- 判据: acc ∈ [55%,80%] 且 >random 0.33

并发 + 落盘 per-task + 断点续跑。OpenAI SDK max_retries=6, timeout=180。
"""
from __future__ import annotations
import json, re, os, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from runtime import data_dir, runs_dir, model_name, make_record_client
import eval_metrics as EM

JST = timezone(timedelta(hours=9))
DATA = str(data_dir())
MODEL = model_name()

# ---- 字段 (与 anchor3_split_assert 一致, 32 真实可见) ----
AGENT_VISIBLE = [
    "admittime","admission_location","admission_type","admission_age","gender","race",
    "insurance","marital_status","language",
    "hypertension","congestive_heart_failure","cerebrovascular_disease",
    "peripheral_vascular_disease","myocardial_infarct","diabetes","smoker","alcohol_abuse",
    "custom_acei","custom_arb","custom_Statins","custom_Metformin","custom_胰岛素",
    "glucose_first","sodium_first","potassium_first","chloride_first","bicarbonate_first",
    "hemoglobin_a1c_first","triglycerides_first","total_cholesterol_first",
    "height","weight",
]
LABEL_FROM_ANSWER = {
    "晚期CKD/ESKD（重度肾衰/透析相关）": "A",
    "CKD（慢性肾病）": "B",
    "AKI（急性肾损伤）": "C",
}
OPTION_TEXT = {
    "A": "晚期CKD/ESKD（重度肾衰/透析相关）",
    "B": "CKD（慢性肾病）",
    "C": "AKI（急性肾损伤）",
}

def parse_fields(q):
    i = q.find("admittime"); b = q[i:] if i >= 0 else q
    d = {}
    for p in re.split(r'[；;]', b):
        p = p.strip()
        if ':' in p:
            k, v = p.split(':', 1); d[k.strip()] = v.strip()
    return d

def load(split):
    return [json.loads(l) for l in open(f"{DATA}/{split}.jsonl", encoding="utf-8")]

SYSTEM = (
    "你是一名临床记录审核助手。你将看到一位患者的入院时可知的结构化病历字段。"
    "任务: 在三个肾脏疾病类别中判断该记录最可能的分类。只能依据给定字段推断, "
    "不得臆造未提供的检验值(如肌酐/eGFR/UACR 本数据不存在)。\n"
    "三个类别:\n"
    "A = 晚期CKD/ESKD（重度肾衰/透析相关）\n"
    "B = CKD（慢性肾病）\n"
    "C = AKI（急性肾损伤）\n"
    "严格输出 JSON: {\"label\": \"A|B|C\", \"evidence_ids\": [使用到的字段名], "
    "\"uncertainty\": 0.0-1.0}。不要输出任何额外文字。"
)

def build_user_prompt(fields):
    lines = []
    for k in AGENT_VISIBLE:
        v = fields.get(k, "NA")
        lines.append(f"{k}: {v}")
    return "该患者可见字段如下:\n" + "\n".join(lines) + "\n\n请给出三分类判断 (JSON)。"

def extract_label(text):
    """从模型输出稳健提取 A/B/C。"""
    if not text:
        return None
    # 优先 JSON
    try:
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).strip().upper()
            if lab in ("A", "B", "C"):
                return lab, obj
    except Exception:
        pass
    # 退化: 找首个独立 A/B/C
    m = re.search(r'\b([ABC])\b', text.upper())
    if m:
        return m.group(1), {}
    return None, {}

def call_one(client, idx, row, rundir, effort):
    fpath = f"{rundir}/tasks/{idx:04d}.json"
    if os.path.exists(fpath):
        try:
            return idx, json.load(open(fpath, encoding="utf-8"))
        except Exception:
            pass
    fields = parse_fields(row["question"])
    gold = LABEL_FROM_ANSWER[row["answer"]]
    t0 = time.time()
    err = None; content = None; usage = None; pred = None; obj = {}
    for attempt in range(3):
        try:
            kwargs = dict(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_user_prompt(fields)}],
                timeout=180,
            )
            if effort:
                kwargs["reasoning_effort"] = effort
            r = client.chat.completions.create(**kwargs)
            content = r.choices[0].message.content
            usage = r.usage.model_dump() if r.usage else None
            pred, obj = extract_label(content)
            if pred:
                break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(2 * (attempt + 1))
    rec = {
        "idx": idx, "gold": gold, "pred": pred,
        "evidence_ids": obj.get("evidence_ids", []),
        "uncertainty": obj.get("uncertainty", None),
        "raw": (content or "")[:500],
        "error": err if not pred else None,
        "latency_s": round(time.time() - t0, 2),
        "usage": usage,
        "model": MODEL, "subject_id": fields.get("subject_id"),
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return idx, rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0=全部 270")
    ap.add_argument("--effort", type=str, default="", help="reasoning_effort: ''|high|default")
    ap.add_argument("--rundir", type=str, default=str(runs_dir() / 'anchor1'))
    args = ap.parse_args()

    rundir = args.rundir
    os.makedirs(f"{rundir}/tasks", exist_ok=True)
    effort = args.effort.strip()

    test = load("test")
    if args.limit > 0:
        test = test[:args.limit]
    client = make_record_client()

    print(f"[{datetime.now(JST).isoformat()}] Autonomous Agent start: n={len(test)} workers={args.workers} model={MODEL} effort={effort or 'default'} rundir={rundir}")
    records = [None] * len(test)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_one, client, i, row, rundir, effort): i for i, row in enumerate(test)}
        for fut in as_completed(futs):
            idx, rec = fut.result()
            records[idx] = rec
            done += 1
            if done % 20 == 0 or done == len(test):
                print(f"  progress {done}/{len(test)}  (last idx={idx} pred={rec['pred']} gold={rec['gold']})", flush=True)

    # 评估
    preds = [r["pred"] if r["pred"] else "A" for r in records]  # 无法解析的记为多数类 A (保守, 不丢样本)
    golds = [r["gold"] for r in records]
    subjects = [r["subject_id"] for r in records]
    n_parse_fail = sum(1 for r in records if not r["pred"])
    n_api_err = sum(1 for r in records if r["error"])

    acc = EM.record_accuracy(preds, golds)
    # bootstrap CI for overall acc (cluster by subject)
    correct = [1.0 if p == g else 0.0 for p, g in zip(preds, golds)]
    acc_ci = EM.cluster_bootstrap_ci(correct, subjects, n_boot=10000, seed=42)

    total_tokens = sum((r["usage"]["total_tokens"] if r.get("usage") else 0) for r in records)

    result = {
        "project_id": "humanchoice",
        "experiment_id": "anchor_1",
        "anchor": "H0.anchor_1 + lit_anchor1_range + eda_akirecall",
        "timestamp": datetime.now(JST).isoformat(),
        "method": "Autonomous Agent (gpt-5.6-sol, fully autonomous, 32 visible fields)",
        "description": "固定后端三分类复现, 命门; 仅可见字段, 无泄漏",
        "config": {"model": MODEL, "n_test": len(test), "visible_fields": len(AGENT_VISIBLE),
                   "reasoning_effort": effort or "default", "workers": args.workers},
        "reference": {"clean_logreg_acc": 71.48, "clean_logreg_aki_recall": 40.5,
                      "majority_acc": 60.74, "random_acc": 33.3,
                      "range_pass": [55.0, 80.0]},
        "results": {
            "overall_acc": round(acc["overall_acc"] * 100, 2),
            "overall_acc_ci95": [round(acc_ci["ci_lo"] * 100, 2), round(acc_ci["ci_hi"] * 100, 2)],
            "macro_f1": round(acc["macro_f1"] * 100, 2),
            "per_class_recall": {k: (round(v * 100, 1) if v is not None else None)
                                 for k, v in acc["per_class_recall"].items()},
            "aki_recall": round(acc["aki_recall"] * 100, 1),
            "support": acc["support"],
            "n_parse_fail": n_parse_fail,
            "n_api_err": n_api_err,
        },
        "cost": {"total_tokens": total_tokens},
        "judgment": {
            "acc_in_range_55_80": 55.0 <= acc["overall_acc"] * 100 <= 80.0,
            "acc_above_random": acc["overall_acc"] * 100 > 33.3,
            "note": "acc 在区间内但 AKI recall 低 ≠ 失败, 是 Augmentation 目标信号; 仅 acc 崩溃(<55 或 ≈random)才算复现失败",
        },
    }
    with open(f"{rundir}/anchor1_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))
    print(f"judgment: {result['judgment']}")
    print(f"saved {rundir}/anchor1_result.json")

if __name__ == "__main__":
    main()
