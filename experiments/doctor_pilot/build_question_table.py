#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a blank question table from selected records.

Copy each record's allowed evidence into the table and leave the question,
options, and response columns empty. This script builds the skeleton only;
the script used to fill the historical question table was not saved.
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
# Missing kidney measurements are recorded explicitly; Unresolved is allowed.
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
    """Attach a source ID and the original value to each allowed evidence field."""
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
            "checkpoint": 0,                     # Start the pilot with a single decision checkpoint.
            "source_record_id": f"test_row_{idx}",
            "subject_id": c["subject_id"],
            "pilot_reason": c["pilot_reason"],
            "gold_label_HIDDEN": c["gold_label"],   # Reference fields are hidden from the participant.
            # The question and options will be filled by a separate generation step.
            "question": "[PENDING_S2]",
            "question_type": "[PENDING_S2]",       # scope/rule/evidence/temporal/preference
            "option_c1": "[PENDING_S2]",
            "option_c2": "[PENDING_S2]",
            "option_c3": "[PENDING_S2]",
            "other_options": "Insufficient Evidence | None of the Above | Skip",
            # Evidence copied from the local record.
            "evidence_spans_json": json.dumps(spans, ensure_ascii=False),
            "n_evidence_fields": len(spans),
            "n_missing_fields": n_missing,
            "renal_goldstandard_absent": "|".join(RENAL_GOLD_ABSENT),
            "reference_allows_unresolved": "TRUE" if c["pilot_reason"].startswith("evidence_scarce") or c["gold_label"] in ("A","B") else "CHECK",
            # Empty columns for the participant's reply.
            "doctor_answer": "",
            "doctor_seconds": "",
            # Mark which parts of the table have been filled.
            "s2_status": "PENDING",
        }
        table.append(row)
    # Save the blank table as CSV and JSON.
    csv_path = f"{OUT}/question_table_skeleton.csv"
    cols = [k for k in table[0].keys() if k != "evidence_spans_json"] + ["evidence_spans_json"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in table: w.writerow(r)
    with open(f"{OUT}/question_table_skeleton.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    meta = {
        "timestamp": datetime.now(JST).isoformat(),
        "stage": "Blank table built locally without model calls",
        "n_rows": len(table),
        "pending_s2_fields": ["question", "question_type", "option_c1/c2/c3"],
        "note": "Question text and answer options are filled by a separate generation step.",
        "outputs": {"csv": csv_path, "json": f"{OUT}/question_table_skeleton.json"},
    }
    with open(f"{OUT}/question_table_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nsaved {csv_path} ({len(table)} rows; evidence copied, questions still empty)")

def fill():
    raise SystemExit("The historical fill runner is unavailable. Use --stage skeleton for the local template.")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["skeleton", "fill"], default="skeleton", help='Use skeleton to build a blank table; the historical fill runner is unavailable')
    args = ap.parse_args()
    if args.stage == "skeleton":
        skeleton()
    else:
        fill()

if __name__ == "__main__":
    main()
