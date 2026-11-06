#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_question_table.py — 医生题表组装 (骨架; S2 候选题由研究执行器生成)

两阶段:
  阶段1 (本地, 现在可做): 读 pilot_candidates + 每患者的 agent_visible 证据, 生成题表骨架
                          (每患者一行占位, 证据 spans 填好, 候选题列留空 [PENDING_S2])
  阶段2 (候选题由研究执行器生成): 调 S2 候选题生成 (pipeline.py 的 gen_questions), 填入 substantive_options
                          → 输出最终医生题表 (CSV + markdown, 对齐 exp_design §5.2)

本脚本实现阶段1 (--stage skeleton)；历史阶段2的专用运行器未随代码保存。

题表字段 (exp_design §5.2):
  task_id | checkpoint | source_record_id | question(题干) | question_type |
  evidence(spans, agent_visible) | option_c1/c2[/c3] | other(insufficient/none/skip) |
  [doctor_answer:医生填] | [doctor_seconds:系统计时]
"""
import json, re, os, csv, argparse, hashlib
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import data_dir, runs_dir

DATA = str(data_dir())
PILOT = str(runs_dir() / 'doctor_pilot')
OUT = PILOT

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
# 肾病金标准 — 恒 ABSENT, 题表须标注"证据缺失", reference 允许 Unresolved
RENAL_GOLD_ABSENT = ["creatinine", "eGFR", "UACR", "creatinine_trajectory"]

def parse_fields(q):
    i = q.find("admittime"); b = q[i:] if i >= 0 else q
    d = {}
    for p in re.split(r'[；;]', b):
        p = p.strip()
        if ':' in p:
            k, v = p.split(':', 1); d[k.strip()] = v.strip()
    return d

def miss(v): return v is None or str(v).strip().upper() in ("", "NA")

def load(split):
    return [json.loads(l) for l in open(f"{DATA}/{split}.jsonl", encoding="utf-8")]

def evidence_spans(fields):
    """把 agent_visible 字段整理成 evidence spans (source_id=字段名, value=原值/NA)。"""
    spans = []
    for k in AGENT_VISIBLE:
        v = fields.get(k, "NA")
        spans.append({"source_id": k, "value": v, "missing": miss(v)})
    return spans

def skeleton():
    pilot = json.load(open(f"{PILOT}/pilot_candidates.json", encoding="utf-8"))["candidates"]
    test_rows = load("test")
    table = []
    for c in pilot:
        idx = c["row_idx"]
        fields = parse_fields(test_rows[idx]["question"])
        task_id = f"med_pilot_{c['subject_id']}"
        spans = evidence_spans(fields)
        n_missing = sum(1 for s in spans if s["missing"])
        row = {
            "task_id": task_id,
            "checkpoint": 0,                     # Pilot 单 checkpoint 起步
            "source_record_id": f"test_row_{idx}",
            "subject_id": c["subject_id"],
            "pilot_reason": c["pilot_reason"],
            "gold_label_HIDDEN": c["gold_label"],   # 医生不可见, 仅 evaluation_only
            # 题干与选项待 S2 生成 (需 GPT)
            "question": "[PENDING_S2]",
            "question_type": "[PENDING_S2]",       # scope/rule/evidence/temporal/preference
            "option_c1": "[PENDING_S2]",
            "option_c2": "[PENDING_S2]",
            "option_c3": "[PENDING_S2]",
            "other_options": "Insufficient Evidence | None of the Above | Skip",
            # 证据 (本地已填好)
            "evidence_spans_json": json.dumps(spans, ensure_ascii=False),
            "n_evidence_fields": len(spans),
            "n_missing_fields": n_missing,
            "renal_goldstandard_absent": "|".join(RENAL_GOLD_ABSENT),
            "reference_allows_unresolved": "TRUE" if c["pilot_reason"].startswith("evidence_scarce") or c["gold_label"] in ("A","B") else "CHECK",
            # 医生填写列 (空)
            "doctor_answer": "",
            "doctor_seconds": "",
            # 生成标记
            "s2_status": "PENDING",
        }
        table.append(row)
    # 写骨架 CSV + JSON
    csv_path = f"{OUT}/question_table_skeleton.csv"
    cols = [k for k in table[0].keys() if k != "evidence_spans_json"] + ["evidence_spans_json"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in table: w.writerow(r)
    with open(f"{OUT}/question_table_skeleton.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    meta = {
        "timestamp": datetime.now(JST).isoformat(),
        "stage": "skeleton (阶段1, 本地无GPT)",
        "n_rows": len(table),
        "pending_s2_fields": ["question", "question_type", "option_c1/c2/c3"],
        "note": "候选题(question/options)需外部 S2 候选题生成",
        "outputs": {"csv": csv_path, "json": f"{OUT}/question_table_skeleton.json"},
    }
    with open(f"{OUT}/question_table_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nsaved {csv_path} ({len(table)} rows, 证据已填, 候选题待 S2)")

def fill():
    raise SystemExit("The historical fill runner is unavailable. Use --stage skeleton for the local template.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["skeleton", "fill"], default="skeleton")
    args = ap.parse_args()
    if args.stage == "skeleton":
        skeleton()
    else:
        fill()

if __name__ == "__main__":
    main()
