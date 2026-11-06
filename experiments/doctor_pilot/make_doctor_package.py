#!/usr/bin/env python3
"""Build doctor fill-in package: (1) blank form (gold hidden) + (2) answer key (gold).
No modeling. Pure reformat of question_table_FINAL.json.
"""
import argparse, json, csv, html, os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="Local, restricted question-table JSON")
parser.add_argument("--output-dir", required=True, help="Local directory for restricted forms")
args = parser.parse_args()
HERE = args.output_dir
os.makedirs(HERE, exist_ok=True)
with open(args.input, encoding="utf-8") as source:
    rows = json.load(source)
assert isinstance(rows, list), "expected list"

GOLD_NAME = {"A": "A = ESKD / 长期维持性透析", "B": "B = 非ESKD 慢性CKD(≥3月)", "C": "C = 新发 AKI"}
OPT_LABELS = ["C1", "C2", "C3", "Insufficient Evidence", "None of the Above", "Skip"]

def ev_lines(ev_json):
    try:
        ev = json.loads(ev_json)
    except Exception:
        return []
    out = []
    for e in ev:
        sid = e.get("source_id", "")
        if e.get("missing"):
            out.append(f"  - {sid}: (缺失/未记录)")
        else:
            v = str(e.get("value", "")).strip()
            out.append(f"  - {sid}: {v}")
    return out

# ---------- 1) BLANK FORM (gold hidden) ----------
form_md = []
form_md.append("# 医生填写表 — HumanChoice 肾病严重度分级 Pilot\n")
form_md.append("> 共 50 位患者 × 3 题 = 150 题。每题请从 6 个选项中**选一个**，并（可选）填写**用时(秒)**与**简短理由**。\n")
form_md.append("> 选项说明：**C1/C2/C3** 为该题的 3 个实质临床判断；若证据不足请选 **Insufficient Evidence**；若都不对选 **None of the Above**；完全无法判断选 **Skip**。\n")
form_md.append("> 你只能看到病历中实际存在的证据字段（肌酐/eGFR 等可能缺失，这是任务本身的设定）。请基于临床经验作答。\n")
form_md.append("\n---\n")

key_rows = []
form_csv_rows = []

last_subj = None
for i, r in enumerate(rows, 1):
    subj = r.get("subject_id", "")
    if subj != last_subj:
        form_md.append(f"\n## 患者 {subj}  (task {r.get('source_record_id','')})\n")
        last_subj = subj
    q = r.get("question", "")
    qt = r.get("question_type", "")
    c1 = r.get("option_c1", ""); c2 = r.get("option_c2", ""); c3 = r.get("option_c3", "")
    form_md.append(f"\n### 题 {i}  [{qt}]\n")
    form_md.append(f"**问题**: {q}\n")
    form_md.append("\n**证据字段**:\n")
    form_md.extend([l + "\n" for l in ev_lines(r.get("evidence_spans_json", "[]"))])
    form_md.append("\n**选项**:\n")
    form_md.append(f"- [ ] **C1**: {c1}\n")
    form_md.append(f"- [ ] **C2**: {c2}\n")
    form_md.append(f"- [ ] **C3**: {c3}\n")
    form_md.append(f"- [ ] **Insufficient Evidence**（证据不足以判断）\n")
    form_md.append(f"- [ ] **None of the Above**（以上都不对）\n")
    form_md.append(f"- [ ] **Skip**（跳过）\n")
    form_md.append(f"\n**你的答案**: ______    **用时(秒)**: ______    **理由(可选)**: ______________________\n")

    # CSV blank form: one row per question, answer/time blank
    form_csv_rows.append({
        "q_no": i, "subject_id": subj, "task": r.get("source_record_id", ""),
        "checkpoint": r.get("checkpoint", ""), "question_type": qt,
        "question": q, "C1": c1, "C2": c2, "C3": c3,
        "option_4": "Insufficient Evidence", "option_5": "None of the Above", "option_6": "Skip",
        "DOCTOR_ANSWER": "", "TIME_SECONDS": "", "REASON": "",
    })
    # answer key
    key_rows.append({
        "q_no": i, "subject_id": subj, "task": r.get("source_record_id", ""),
        "question_type": qt, "gold_label": r.get("gold_label_HIDDEN", ""),
        "gold_meaning": GOLD_NAME.get(r.get("gold_label_HIDDEN", ""), ""),
        "pilot_reason": r.get("pilot_reason", ""),
        "question": q, "C1": c1, "C2": c2, "C3": c3,
        "note": "C1 supports A, C2 supports B, C3 supports C (per option text). "
                "正确可为 Insufficient 若该字段在证据中确实缺失。",
    })

open(os.path.join(HERE, "DOCTOR_FORM_BLANK.md"), "w", encoding="utf-8").write("".join(form_md))

with open(os.path.join(HERE, "DOCTOR_FORM_BLANK.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(form_csv_rows[0].keys()))
    w.writeheader(); w.writerows(form_csv_rows)

with open(os.path.join(HERE, "ANSWER_KEY.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
    w.writeheader(); w.writerows(key_rows)

# answer key md (compact)
km = ["# 答案 Key — HumanChoice Pilot (保密, 勿给医生)\n",
      f"> 共 {len(key_rows)} 题。gold_label 为该患者真实肾病严重度分级标签。\n",
      "> 标签含义: A=ESKD/透析, B=非ESKD CKD(≥3月), C=新发AKI。\n",
      "> 选项映射: C1→支持A, C2→支持B, C3→支持C。若证据字段确实缺失, Insufficient 也算正确。\n\n",
      "| 题# | 患者 | task | 类型 | gold | 含义 | pilot_reason |\n",
      "|---|---|---|---|---|---|---|\n"]
for k in key_rows:
    km.append(f"| {k['q_no']} | {k['subject_id']} | {k['task']} | {k['question_type']} | "
              f"**{k['gold_label']}** | {k['gold_meaning']} | {k['pilot_reason']} |\n")
open(os.path.join(HERE, "ANSWER_KEY.md"), "w", encoding="utf-8").write("".join(km))

# distribution summary
from collections import Counter
dist = Counter(k["gold_label"] for k in key_rows)
reason = Counter(k["pilot_reason"] for k in key_rows)
print("questions:", len(key_rows))
print("patients:", len({k["subject_id"] for k in key_rows}))
print("gold dist:", dict(dist))
print("pilot_reason:", dict(reason))
print("files: DOCTOR_FORM_BLANK.md, DOCTOR_FORM_BLANK.csv, ANSWER_KEY.csv, ANSWER_KEY.md")
