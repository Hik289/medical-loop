#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize the fields in the saved medical records.

Parse the fields stored in each question. Count missing values and labels,
check patient separation, and identify fields that must be hidden from the
agent because they contain answers or later outcomes.
"""
import json, re, os, statistics
from collections import Counter, defaultdict

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import data_dir, runs_dir

DATA_DIR = str(data_dir())
OUT_DIR  = os.path.join(DATA_DIR, "eda_out")
os.makedirs(OUT_DIR, exist_ok=True)

def load(split):
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8")]

# ---- parse embedded fields from the question string ----
# The question ends with structured key:value fields separated by semicolons.
# separator is the full-width semicolon '；'; key:value by first ':' (half-width).
PREAMBLE_SPLIT = "："  # The record header uses a full-width colon.

def parse_fields(q):
    # locate the structured block: after the last full-width colon before first field
    # Fields begin at "admittime:" typically. Find that anchor.
    idx = q.find("admittime")
    block = q[idx:] if idx >= 0 else q
    parts = block.split("；")
    d = {}
    order = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # split on first half-width colon
        if ":" in p:
            k, v = p.split(":", 1)
        elif "：" in p:
            k, v = p.split("：", 1)
        else:
            continue
        k = k.strip()
        v = v.strip()
        if k and k not in d:
            order.append(k)
        d[k] = v
    return d, order

def is_missing(v):
    return v is None or v == "" or str(v).strip().upper() == "NA"

def try_num(v):
    if is_missing(v):
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ---- LEAKAGE CLASSIFICATION ----
# Answer fields (not in question but in record JSON): answer, answer_idx, options, meta_info
# Outcome fields (unknowable at admission / result-type) per HumanLoop 12.3 step2:
OUTCOME_FIELDS = {
    "dischtime", "discharge_location", "los_hospital",
    "hospital_expire_flag", "icu_expire_flag",
    "follow_up_years", "survival_days",
    "mortality_28", "mortality_30", "mortality_90", "mortality_365",
    "los_icu", "careunit", "stay_id", "hospstay_seq",
}
# ID fields: keep as Source IDs for linkage but NOT as clinical evidence to agent
ID_FIELDS = {"subject_id", "hadm_id"}
# sofa is an ICU severity score computed during stay -> treat as potential outcome/stay-derived
# flagged separately for manual review
FLAG_FIELDS = {"sofa"}

def main():
    report = {}
    all_rows = {}
    parsed = {}
    orders = {}
    for split in ("train", "test"):
        rows = load(split)
        all_rows[split] = rows
        parsed[split] = []
        for r in rows:
            d, order = parse_fields(r["question"])
            parsed[split].append(d)
            orders.setdefault(split, Counter()).update([tuple(order)])

    # 1. Field inventory (union + per-split presence)
    field_union = []
    seen = set()
    for split in ("train", "test"):
        for d in parsed[split]:
            for k in d.keys():
                if k not in seen:
                    seen.add(k); field_union.append(k)

    # field count consistency per record
    fcount = {s: Counter(len(d) for d in parsed[s]) for s in ("train","test")}

    # 2. Per-field type + missingness
    field_profile = {}
    for k in field_union:
        prof = {"field": k}
        for split in ("train","test"):
            vals = [d.get(k) for d in parsed[split]]
            present = [v for v in vals if v is not None]
            n = len(parsed[split])
            miss = sum(1 for v in vals if is_missing(v))
            nums = [try_num(v) for v in vals]
            nums = [x for x in nums if x is not None]
            # distinct non-missing values (cap sample)
            nonmiss_vals = [str(v) for v in vals if not is_missing(v)]
            distinct = Counter(nonmiss_vals)
            is_numeric = len(nums) > 0 and len(nums) >= 0.8*max(1,len(nonmiss_vals))
            entry = {
                "n": n,
                "missing": miss,
                "missing_rate": round(miss/n, 4) if n else None,
                "n_distinct": len(distinct),
                "numeric": is_numeric,
            }
            if is_numeric and nums:
                entry["num_stats"] = {
                    "mean": round(statistics.mean(nums),4),
                    "median": round(statistics.median(nums),4),
                    "min": round(min(nums),4),
                    "max": round(max(nums),4),
                    "std": round(statistics.pstdev(nums),4) if len(nums)>1 else 0.0,
                }
            else:
                # categorical top values
                entry["top_values"] = distinct.most_common(10)
            prof[split] = entry
        field_profile[k] = prof

    # 3. Label distribution + imbalance
    def label_dist(rows):
        c = Counter(r["answer"] for r in rows)
        idx = Counter(r["answer_idx"][0] for r in rows)
        total = len(rows)
        return {
            "by_answer": {k: {"n": v, "pct": round(v/total,4)} for k,v in c.items()},
            "by_idx":    {k: {"n": v, "pct": round(v/total,4)} for k,v in idx.items()},
            "total": total,
        }
    labels = {s: label_dist(all_rows[s]) for s in ("train","test")}
    # imbalance ratio = majority/minority
    for s in ("train","test"):
        counts = [v["n"] for v in labels[s]["by_idx"].values()]
        labels[s]["imbalance_ratio_maj_min"] = round(max(counts)/min(counts),3)

    # 4. Patient isolation: subject_id / hadm_id across splits
    def ids(split, key):
        out = []
        for d in parsed[split]:
            v = d.get(key)
            if not is_missing(v):
                out.append(str(v).strip())
        return out
    iso = {}
    for key in ("subject_id","hadm_id"):
        tr = set(ids("train",key)); te = set(ids("test",key))
        tr_list = ids("train",key); te_list = ids("test",key)
        overlap = tr & te
        iso[key] = {
            "train_unique": len(tr),
            "train_total": len(tr_list),
            "train_dup_within": len(tr_list)-len(tr),
            "test_unique": len(te),
            "test_total": len(te_list),
            "test_dup_within": len(te_list)-len(te),
            "cross_split_overlap_count": len(overlap),
            "cross_split_overlap_sample": sorted(overlap)[:20],
        }

    # subject->hadm mapping (one patient multiple admissions?)
    subj_hadm = defaultdict(set)
    for split in ("train","test"):
        for d in parsed[split]:
            s = d.get("subject_id"); h = d.get("hadm_id")
            if not is_missing(s) and not is_missing(h):
                subj_hadm[str(s)].add(str(h))
    multi_adm = {s:sorted(h) for s,h in subj_hadm.items() if len(h)>1}

    # 5. Leakage classification
    visible, hidden = [], []
    for k in field_union:
        if k in OUTCOME_FIELDS or k in FLAG_FIELDS:
            hidden.append(k)
        else:
            visible.append(k)
    leakage = {
        "answer_level_fields_in_record_json": ["answer","answer_idx","options","meta_info"],
        "outcome_fields_hidden": sorted(OUTCOME_FIELDS),
        "flagged_for_review": sorted(FLAG_FIELDS),
        "id_fields_keep_as_source_not_evidence": sorted(ID_FIELDS),
        "agent_visible_fields": [k for k in field_union if k in set(visible)],
        "agent_hidden_fields": [k for k in field_union if k in set(hidden)],
        "n_visible": len(visible),
        "n_hidden": len(hidden),
    }

    # 6. key clinical quantities availability (creatinine/eGFR/UACR per HumanLoop)
    key_clinical = ["creatinine","egfr","uacr","gfr","acr","hemoglobin_a1c_first",
                    "glucose_first","sodium_first","potassium_first","chloride_first",
                    "bicarbonate_first","triglycerides_first","total_cholesterol_first"]
    clin_avail = {}
    for name in key_clinical:
        matches = [k for k in field_union if name.lower() in k.lower()]
        for m in matches:
            clin_avail[m] = {
                s: field_profile[m][s]["missing_rate"] for s in ("train","test")
            }
        if not matches:
            clin_avail[name] = "ABSENT_FROM_DATA"

    report = {
        "n_records": {s: len(all_rows[s]) for s in ("train","test")},
        "field_count_per_record": {s: dict(fcount[s]) for s in ("train","test")},
        "field_order_consistency": {s: len(orders[s]) for s in ("train","test")},
        "n_fields_union": len(field_union),
        "field_union": field_union,
        "labels": labels,
        "patient_isolation": iso,
        "patients_with_multiple_admissions": {
            "count": len(multi_adm),
            "sample": dict(list(multi_adm.items())[:10]),
        },
        "leakage_classification": leakage,
        "key_clinical_availability": clin_avail,
    }

    with open(os.path.join(OUT_DIR,"eda_report.json"),"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    with open(os.path.join(OUT_DIR,"field_profile.json"),"w",encoding="utf-8") as f:
        json.dump(field_profile,f,ensure_ascii=False,indent=2)

    # console summary
    print("=== N records ===", report["n_records"])
    print("=== field count/record ===", report["field_count_per_record"])
    print("=== n fields union ===", report["n_fields_union"])
    print("=== labels train ===", json.dumps(labels["train"],ensure_ascii=False))
    print("=== labels test  ===", json.dumps(labels["test"],ensure_ascii=False))
    print("=== isolation ===", json.dumps(iso,ensure_ascii=False))
    print("=== multi-adm patients ===", len(multi_adm))
    print("=== visible/hidden ===", leakage["n_visible"], leakage["n_hidden"])
    print("=== key clinical ===", json.dumps(clin_avail,ensure_ascii=False))
    print("OUT:", OUT_DIR)

if __name__ == "__main__":
    main()
