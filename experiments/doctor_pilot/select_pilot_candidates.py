#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select records for a pilot without calling a model.

Include AKI records and records with many missing laboratory values. Save
the selected records for question generation and form preparation. Selection
uses existing labels and missing-value patterns; it does not verify diagnoses.
"""
import json, re, os, csv
from collections import Counter
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import data_dir, runs_dir

DATA = str(data_dir())
OUT = str(runs_dir() / 'doctor_pilot')
os.makedirs(OUT, exist_ok=True)

LABEL = {"晚期CKD/ESKD（重度肾衰/透析相关）": "A", "CKD（慢性肾病）": "B", "AKI（急性肾损伤）": "C"}
LABEL_NAME = {"A": "晚期CKD/ESKD", "B": "CKD", "C": "AKI"}
# First recorded laboratory values used to identify records with missing evidence.
LABS = ["glucose_first", "sodium_first", "potassium_first", "chloride_first",
        "bicarbonate_first", "hemoglobin_a1c_first", "triglycerides_first", "total_cholesterol_first"]
# Recorded conditions provide indirect evidence.
COMORB = ["hypertension", "congestive_heart_failure", "cerebrovascular_disease",
          "peripheral_vascular_disease", "myocardial_infarct", "diabetes"]

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

def analyze(split):
    rows = load(split)
    cands = []
    for idx, r in enumerate(rows):
        f = parse_fields(r["question"])
        lab = LABEL[r["answer"]]
        lab_missing = sum(miss(f.get(k)) for k in LABS)
        comorb_n = sum(1 for k in COMORB if str(f.get(k, "0")).strip() == "1")
        cands.append({
            "split": split, "row_idx": idx,
            "subject_id": f.get("subject_id"), "hadm_id": f.get("hadm_id"),
            "gold_label": lab, "gold_name": LABEL_NAME[lab],
            "admission_age": f.get("admission_age"), "admission_type": f.get("admission_type"),
            "lab_missing_count": lab_missing, "comorbidity_count": comorb_n,
            # Creatinine, eGFR, and UACR are absent from this dataset.
            "renal_goldstandard_absent": True,
        })
    return cands

def main():
    test_c = analyze("test")
    train_c = analyze("train")

    # Include all 37 AKI test records.
    p1_aki = [c for c in test_c if c["gold_label"] == "C"]

    # Include class A/B records missing at least four laboratory fields.
    # Required kidney measurements are absent, so the reference should allow Unresolved.
    p2_scarce_test = [c for c in test_c if c["gold_label"] in ("A", "B") and c["lab_missing_count"] >= 4]
    # Also include any record missing all eight laboratory fields.
    p2_allmiss_test = [c for c in test_c if c["lab_missing_count"] == 8]

    # Remove duplicate patients from the selected records.
    pilot = {}
    for c in p1_aki:
        pilot[c["subject_id"]] = {**c, "pilot_reason": "AKI_minority"}
    for c in p2_scarce_test:
        if c["subject_id"] not in pilot:
            pilot[c["subject_id"]] = {**c, "pilot_reason": "evidence_scarce_labmiss>=4"}
    for c in p2_allmiss_test:
        if c["subject_id"] not in pilot:
            pilot[c["subject_id"]] = {**c, "pilot_reason": "evidence_scarce_all8_labs_missing"}
    pilot_list = list(pilot.values())

    # Save the selected records as CSV and JSON.
    csv_path = f"{OUT}/pilot_candidates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pilot_list[0].keys()))
        w.writeheader()
        for c in pilot_list: w.writerow(c)

    summary = {
        "timestamp": datetime.now(JST).isoformat(),
        "purpose": "Records selected for preparing pilot questions",
        "selection_criteria": {
            "P1_AKI_minority": "All AKI (class C) test records",
            "P2_evidence_scarce": "Class A/B records missing at least four laboratory values, or any record missing all eight",
            "note": "Creatinine, eGFR, and UACR are absent, so questions requiring them should allow Unresolved.",
        },
        "counts": {
            "test_total": len(test_c),
            "test_label_dist": dict(Counter(c["gold_label"] for c in test_c)),
            "P1_AKI_all": len(p1_aki),
            "P2_scarce_AB_labmiss>=4": len(p2_scarce_test),
            "P2_all8_labs_missing": len(p2_allmiss_test),
            "pilot_unique_patients": len(pilot_list),
            "pilot_by_reason": dict(Counter(c["pilot_reason"] for c in pilot_list)),
            "pilot_by_label": dict(Counter(c["gold_label"] for c in pilot_list)),
        },
        "test_labmiss_distribution": dict(Counter(c["lab_missing_count"] for c in test_c)),
        "outputs": {"csv": csv_path, "json_list": f"{OUT}/pilot_candidates.json"},
    }
    with open(f"{OUT}/pilot_candidates.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "candidates": pilot_list}, f, ensure_ascii=False, indent=2)
    with open(f"{OUT}/pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsaved {csv_path} + pilot_candidates.json + pilot_summary.json")

if __name__ == "__main__":
    main()
