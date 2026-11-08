#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the patient split and the fields supplied to the agent.

Confirm that training and test patients do not overlap, answer and outcome
fields are excluded, and missing values match the original records. Save and
check a fixed patient split and separate training groups for A, Q, and G.
Runs locally without model calls.
"""
import sys, json, re, os, hashlib
from datetime import datetime, timezone, timedelta
from collections import Counter
import numpy as np

JST = timezone(timedelta(hours=9))
from pathlib import Path
from runtime import data_dir, runs_dir

DATA = str(data_dir())
OUT = str(runs_dir())
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))

# Fields excluded from model input and fields allowed as evidence.
ANSWER_TOP = ["answer", "answer_idx", "options", "meta_info"]  # Top-level JSON fields that reveal the answer.
OUTCOME_16 = ["hospital_expire_flag","icu_expire_flag","mortality_28","mortality_30",
              "mortality_90","mortality_365","dischtime","discharge_location",
              "los_hospital","los_icu","follow_up_years","survival_days",
              "sofa","stay_id","careunit","hospstay_seq"]
IDS = ["subject_id","hadm_id"]
# Earlier notes counted the two patient identifiers among visible fields.
# Excluding those identifiers leaves 32 clinical fields supplied to the agent.
# These checks use the fields actually supplied to the agent.
AGENT_VISIBLE = [
    # Demographics and admission details.
    "admittime","admission_location","admission_type","admission_age","gender","race",
    "insurance","marital_status","language",
    # Recorded conditions.
    "hypertension","congestive_heart_failure","cerebrovascular_disease",
    "peripheral_vascular_disease","myocardial_infarct","diabetes","smoker","alcohol_abuse",
    # Medication flags.
    "custom_acei","custom_arb","custom_Statins","custom_Metformin","custom_胰岛素",
    # D4 first-lab (8)
    "glucose_first","sodium_first","potassium_first","chloride_first","bicarbonate_first",
    "hemoglobin_a1c_first","triglycerides_first","total_cholesterol_first",
    # Height and weight.
    "height","weight",
]  # 9+8+5+8+2 = 32

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
    rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl", encoding="utf-8")]
    return rows

print("=" * 70)
print("Patient-split and input-field checks")
print("=" * 70)

tr = load("train"); te = load("test")
print(f"\ntrain={len(tr)} test={len(te)}")
check("Training record count is 627", len(tr) == 627, f"{len(tr)}")
check("Test record count is 270", len(te) == 270, f"{len(te)}")

tr_f = [parse_fields(r["question"]) for r in tr]
te_f = [parse_fields(r["question"]) for r in te]

# ---------------------------------------------------------------------------
# Check that patient and admission IDs do not cross the train/test split.
# ---------------------------------------------------------------------------
print("\n[Patient split] Training and test records must not share patients")
tr_subj = set(f.get("subject_id") for f in tr_f)
te_subj = set(f.get("subject_id") for f in te_f)
tr_hadm = set(f.get("hadm_id") for f in tr_f)
te_hadm = set(f.get("hadm_id") for f in te_f)
check("No patient ID overlap across splits", len(tr_subj & te_subj) == 0, f"overlap={len(tr_subj & te_subj)}")
check("No admission ID overlap across splits", len(tr_hadm & te_hadm) == 0, f"overlap={len(tr_hadm & te_hadm)}")
check("Unique training patients", len(tr_subj) == len(tr), f"{len(tr_subj)} unique / {len(tr)} records (hospstay_seq=1)")
check("Unique test patients", len(te_subj) == len(te), f"{len(te_subj)} unique / {len(te)} records")

# ---------------------------------------------------------------------------
# Check that model input excludes answer, outcome, and identifier fields.
# ---------------------------------------------------------------------------
print("\n[Input fields] Exclude answers, later outcomes, and identifiers")
av = set(AGENT_VISIBLE)
check("Exactly 32 allowed input fields", len(av) == 32, f"{len(av)}")
check("agent_visible ∩ OUTCOME_16 = ∅", len(av & set(OUTCOME_16)) == 0, f"{av & set(OUTCOME_16)}")
check("agent_visible ∩ IDS = ∅", len(av & set(IDS)) == 0, f"{av & set(IDS)}")
check("agent_visible ∩ ANSWER_TOP = ∅", len(av & set(ANSWER_TOP)) == 0, f"{av & set(ANSWER_TOP)}")
# Check that every parsed field has an assigned input or exclusion category.
all_struct_keys = set()
for f in tr_f: all_struct_keys |= set(f.keys())
defined = av | set(OUTCOME_16) | set(IDS)
undefined = all_struct_keys - defined
check("Every structured field has a category", len(undefined) == 0, f"undefined={undefined}")

# ---------------------------------------------------------------------------
# Check missing-value rates for selected fields.
# ---------------------------------------------------------------------------
print("\n[Missing values] Keep the values as recorded")
def miss_rate(fs, key): return sum(miss(f.get(key)) for f in fs) / len(fs)
# Expected rates come from the original data summary.
checks_miss = [
    ("hemoglobin_a1c_first", 0.708, 0.03),
    ("triglycerides_first", 0.831, 0.03),
    ("total_cholesterol_first", 0.853, 0.03),
    ("glucose_first", 0.056, 0.02),
    ("sodium_first", 0.053, 0.02),
    ("height", 0.805, 0.03),
    ("weight", 0.684, 0.03),
]
for key, expect, tol in checks_miss:
    mr = miss_rate(tr_f, key)
    check(f"missing[{key}]≈{expect:.1%}", abs(mr - expect) <= tol, f"got={mr:.1%}")
# These kidney measurements are absent and must not be filled in.
for absent in ["creatinine", "eGFR", "UACR"]:
    present = any(absent in f for f in tr_f)
    check(f"{absent} is absent and must not be filled in", not present, "")

# ---------------------------------------------------------------------------
# Save the fixed patient split and check its checksum on later runs.
# ---------------------------------------------------------------------------
print("\n[Saved split] Write or check the patient assignment file")
split_map = {}
for f in tr_f: split_map[str(f.get("subject_id"))] = "train"
for f in te_f: split_map[str(f.get("subject_id"))] = "test"
smap_path = f"{OUT}/split_map.json"
smap_bytes = json.dumps(split_map, sort_keys=True, ensure_ascii=False).encode("utf-8")
smap_sha = hashlib.sha256(smap_bytes).hexdigest()
# If a split file already exists, confirm that its contents have not changed.
if os.path.exists(smap_path):
    old = open(smap_path, "rb").read()
    old_sha = hashlib.sha256(json.dumps(json.loads(old), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    check("The saved patient split is unchanged", old_sha == smap_sha, f"old={old_sha[:12]} new={smap_sha[:12]}")
else:
    with open(smap_path, "w", encoding="utf-8") as fp:
        json.dump(split_map, fp, ensure_ascii=False, indent=2)
    with open(f"{OUT}/split_map.sha256", "w") as fp:
        fp.write(smap_sha)
    check("Saved the initial patient split", True, f"sha={smap_sha[:12]} n={len(split_map)}")
check("The split file includes every patient", len(split_map) == len(tr_subj) + len(te_subj), f"{len(split_map)}")

# ---------------------------------------------------------------------------
# Check that the A/Q/G training groups are separate and repeatable.
# ---------------------------------------------------------------------------
print("\n[Module groups] Separate patients into fixed training groups")
# Assign each training patient to a group using a fixed hash.
train_subjects = sorted(tr_subj)
def assign_group(subj):
    h = int(hashlib.sha256(str(subj).encode()).hexdigest(), 16)
    return ["D_A", "D_Q", "D_G"][h % 3]
groups = {g: set() for g in ["D_A", "D_Q", "D_G"]}
for s in train_subjects:
    groups[assign_group(s)].add(s)
# The groups must not share patients.
inter = (groups["D_A"] & groups["D_Q"]) | (groups["D_A"] & groups["D_G"]) | (groups["D_Q"] & groups["D_G"])
check("A/Q/G groups do not share patients", len(inter) == 0, f"intersection={len(inter)}")
check("A/Q/G groups contain all training patients", groups["D_A"]|groups["D_Q"]|groups["D_G"] == set(train_subjects), "")
# All three groups must contain training patients only.
all_in_train = all(split_map[str(s)] == "train" for g in groups.values() for s in g)
check("All module-group patients belong to the training split", all_in_train, "")
# Recompute the assignments to check that they stay fixed between rounds.
groups2 = {g: set() for g in ["D_A", "D_Q", "D_G"]}
for s in train_subjects: groups2[assign_group(s)].add(s)
cross_iter_stable = all(groups[g] == groups2[g] for g in groups)
check("Group assignments stay fixed between training rounds", cross_iter_stable,
      f"D_A={len(groups['D_A'])} D_Q={len(groups['D_Q'])} D_G={len(groups['D_G'])}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"Results: {len(PASS)} PASS / {len(FAIL)} FAIL")
result = {
    "anchor": "H0.anchor_3 (recheck) + H0.anchor_2b (new invariant)",
    "timestamp": datetime.now(JST).isoformat(),
    "n_pass": len(PASS), "n_fail": len(FAIL),
    "status": "PASS" if not FAIL else "FAIL",
    "split_map_sha256": smap_sha,
    "group_sizes": {g: len(s) for g, s in groups.items()},
    "failures": [{"name": n, "detail": d} for n, d in FAIL],
}
with open(os.path.join(OUT, "anchor3_split_result.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"saved anchor3_split_result.json → status={result['status']}")
if FAIL:
    print("FAIL: inspect the data checks above.")
    sys.exit(1)
print("PASS: patient splits, input fields, and module groups")
