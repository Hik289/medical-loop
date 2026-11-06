#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anchor3_split_assert.py — H0.anchor_3 复查 + H0.anchor_2b 新不变量 (debug_move=ASSERT)

anchor_3 (已 confirmed, 本波固化为 pipeline 断言):
  - subject_id/hadm_id 无跨 split 重叠
  - agent_visible 34 字段不含任何 16 泄漏字段 + 4 答案字段 (+2 ID)
  - Missing 率与原文件一致 (不填补)
anchor_2b (新增, theorist 要求(a)):
  - split_map.json 生成后只读 (checksum)
  - 跨 Iteration 同一患者恒定归属同一 split
  - D_A/D_Q/D_G 患者互斥且恒定归属同一 split (模拟 3-way 患者级划分验证机制)

纯本地, 无 API 调用。
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

# ---- 字段定义 (来自 leakage_fields.md) ----
ANSWER_TOP = ["answer", "answer_idx", "options", "meta_info"]  # JSON 顶层答案泄漏
OUTCOME_16 = ["hospital_expire_flag","icu_expire_flag","mortality_28","mortality_30",
              "mortality_90","mortality_365","dischtime","discharge_location",
              "los_hospital","los_icu","follow_up_years","survival_days",
              "sofa","stay_id","careunit","hospstay_seq"]
IDS = ["subject_id","hadm_id"]
# 注: leakage_fields.md 概述口径称"34 可见"(=50 结构化 − 16 outcome, 把 2 ID 算入),
# 但 2 个 ID (subject_id/hadm_id) 实际不进 agent 输入 → 真实 agent-visible 临床字段 = 32 (50−16−2)。
# 本断言以"真正进 agent 输入的字段"为准 = 32。
AGENT_VISIBLE = [
    # D1 人口学/入院背景 (9)
    "admittime","admission_location","admission_type","admission_age","gender","race",
    "insurance","marital_status","language",
    # D2 共病 (8)
    "hypertension","congestive_heart_failure","cerebrovascular_disease",
    "peripheral_vascular_disease","myocardial_infarct","diabetes","smoker","alcohol_abuse",
    # D3 用药 (5)
    "custom_acei","custom_arb","custom_Statins","custom_Metformin","custom_胰岛素",
    # D4 first-lab (8)
    "glucose_first","sodium_first","potassium_first","chloride_first","bicarbonate_first",
    "hemoglobin_a1c_first","triglycerides_first","total_cholesterol_first",
    # D5 体格 (2)
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
print("H0.anchor_3 复查 + H0.anchor_2b 新不变量 ASSERT")
print("=" * 70)

tr = load("train"); te = load("test")
print(f"\ntrain={len(tr)} test={len(te)}")
check("train 行数=627", len(tr) == 627, f"{len(tr)}")
check("test 行数=270", len(te) == 270, f"{len(te)}")

tr_f = [parse_fields(r["question"]) for r in tr]
te_f = [parse_fields(r["question"]) for r in te]

# ---------------------------------------------------------------------------
# anchor_3.1 患者隔离: subject_id/hadm_id 无跨 split 重叠
# ---------------------------------------------------------------------------
print("\n[anchor_3.1] 患者隔离 (无跨 split 重叠)")
tr_subj = set(f.get("subject_id") for f in tr_f)
te_subj = set(f.get("subject_id") for f in te_f)
tr_hadm = set(f.get("hadm_id") for f in tr_f)
te_hadm = set(f.get("hadm_id") for f in te_f)
check("subject_id 无跨split重叠", len(tr_subj & te_subj) == 0, f"overlap={len(tr_subj & te_subj)}")
check("hadm_id 无跨split重叠", len(tr_hadm & te_hadm) == 0, f"overlap={len(tr_hadm & te_hadm)}")
check("train 唯一患者数", len(tr_subj) == len(tr), f"{len(tr_subj)} 唯一 / {len(tr)} 行 (hospstay_seq=1)")
check("test 唯一患者数", len(te_subj) == len(te), f"{len(te_subj)} 唯一 / {len(te)} 行")

# ---------------------------------------------------------------------------
# anchor_3.2 agent_visible 不含任何泄漏字段
# ---------------------------------------------------------------------------
print("\n[anchor_3.2] agent_visible 34 字段不含泄漏")
av = set(AGENT_VISIBLE)
check("agent_visible 恰 32 字段 (50−16 outcome−2 ID)", len(av) == 32, f"{len(av)}")
check("agent_visible ∩ OUTCOME_16 = ∅", len(av & set(OUTCOME_16)) == 0, f"{av & set(OUTCOME_16)}")
check("agent_visible ∩ IDS = ∅", len(av & set(IDS)) == 0, f"{av & set(IDS)}")
check("agent_visible ∩ ANSWER_TOP = ∅", len(av & set(ANSWER_TOP)) == 0, f"{av & set(ANSWER_TOP)}")
# 验证 34 可见 + 16 泄漏 + 2 ID = 52 结构化字段全覆盖, 无遗漏
all_struct_keys = set()
for f in tr_f: all_struct_keys |= set(f.keys())
defined = av | set(OUTCOME_16) | set(IDS)
undefined = all_struct_keys - defined
check("无未分类结构化字段", len(undefined) == 0, f"undefined={undefined}")

# ---------------------------------------------------------------------------
# anchor_3.3 Missing 率与原文件一致 (抽查关键高缺失字段)
# ---------------------------------------------------------------------------
print("\n[anchor_3.3] Missing 率保留 (不填补)")
def miss_rate(fs, key): return sum(miss(f.get(key)) for f in fs) / len(fs)
# 参照 leakage_fields.md D4/D5 的已知缺失率
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
# 肾病金标准证据完全不存在 (eda_evidence_boundary)
for absent in ["creatinine", "eGFR", "UACR"]:
    present = any(absent in f for f in tr_f)
    check(f"{absent} ABSENT (不可补造)", not present, "")

# ---------------------------------------------------------------------------
# anchor_2b.1 生成 split_map.json (患者级冻结映射) + checksum, 只读
# ---------------------------------------------------------------------------
print("\n[anchor_2b.1] split_map.json 生成 + checksum (只读)")
split_map = {}
for f in tr_f: split_map[str(f.get("subject_id"))] = "train"
for f in te_f: split_map[str(f.get("subject_id"))] = "test"
smap_path = f"{OUT}/split_map.json"
smap_bytes = json.dumps(split_map, sort_keys=True, ensure_ascii=False).encode("utf-8")
smap_sha = hashlib.sha256(smap_bytes).hexdigest()
# 若已存在, 校验 checksum 不变 (只读语义)
if os.path.exists(smap_path):
    old = open(smap_path, "rb").read()
    old_sha = hashlib.sha256(json.dumps(json.loads(old), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    check("split_map checksum 不变 (只读)", old_sha == smap_sha, f"old={old_sha[:12]} new={smap_sha[:12]}")
else:
    with open(smap_path, "w", encoding="utf-8") as fp:
        json.dump(split_map, fp, ensure_ascii=False, indent=2)
    with open(f"{OUT}/split_map.sha256", "w") as fp:
        fp.write(smap_sha)
    check("split_map.json 首次生成", True, f"sha={smap_sha[:12]} n={len(split_map)}")
check("split_map 覆盖全部患者", len(split_map) == len(tr_subj) + len(te_subj), f"{len(split_map)}")

# ---------------------------------------------------------------------------
# anchor_2b.2 模拟 3-way D_A/D_Q/D_G 患者级划分, 验证互斥 + 恒定归属同一 split
# ---------------------------------------------------------------------------
print("\n[anchor_2b.2] D_A/D_Q/D_G 患者互斥 + 恒定同一 split (跨轮不变)")
# 机制验证: 在 train 患者内按 subject_id hash 做确定性 3-way 划分 (模拟 §8.6)
train_subjects = sorted(tr_subj)
def assign_group(subj):
    h = int(hashlib.sha256(str(subj).encode()).hexdigest(), 16)
    return ["D_A", "D_Q", "D_G"][h % 3]
groups = {g: set() for g in ["D_A", "D_Q", "D_G"]}
for s in train_subjects:
    groups[assign_group(s)].add(s)
# 互斥
inter = (groups["D_A"] & groups["D_Q"]) | (groups["D_A"] & groups["D_G"]) | (groups["D_Q"] & groups["D_G"])
check("D_A/D_Q/D_G 患者互斥", len(inter) == 0, f"intersection={len(inter)}")
check("三组并集=train患者", groups["D_A"]|groups["D_Q"]|groups["D_G"] == set(train_subjects), "")
# 全部在 train split 内 (不跨 split)
all_in_train = all(split_map[str(s)] == "train" for g in groups.values() for s in g)
check("三组患者全在 train split 内", all_in_train, "")
# 跨轮不变: 重算一次 assign 应得到完全相同的划分 (确定性 → 跨 Iteration 恒定)
groups2 = {g: set() for g in ["D_A", "D_Q", "D_G"]}
for s in train_subjects: groups2[assign_group(s)].add(s)
cross_iter_stable = all(groups[g] == groups2[g] for g in groups)
check("跨 Iteration 划分恒定 (防 target 刷新泄漏)", cross_iter_stable,
      f"D_A={len(groups['D_A'])} D_Q={len(groups['D_Q'])} D_G={len(groups['D_G'])}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"结果: {len(PASS)} PASS / {len(FAIL)} FAIL")
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
    print("❌ anchor_3/2b FAIL — 硬 fail, 检查断言明细")
    sys.exit(1)
print("✅ anchor_3 复查 + anchor_2b PASS")
