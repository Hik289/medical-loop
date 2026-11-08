#!/usr/bin/env python3
"""
Format a local question table as a blank response form and a separate answer key.

The blank form hides reference labels. This script only formats files; it
does not call a model or collect participant responses.
"""
import argparse, json, csv, html, os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="Local JSON question table")
parser.add_argument("--output-dir", required=True, help="Private local folder for the response form and answer key")
args = parser.parse_args()
HERE = args.output_dir
os.makedirs(HERE, exist_ok=True)
with open(args.input, encoding="utf-8") as source:
    rows = json.load(source)
assert isinstance(rows, list), "expected list"

GOLD_NAME = {"A": "A = ESKD / long-term maintenance dialysis", "B": "B = CKD without ESKD, present for at least three months", "C": "C = new AKI"}
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
            out.append(f"  - {sid}: (missing or not recorded)")
        else:
            v = str(e.get("value", "")).strip()
            out.append(f"  - {sid}: {v}")
    return out

# ---------- 1) BLANK FORM (gold hidden) ----------
form_md = []
form_md.append("# HumanChoice pilot response form\n")
form_md.append("> Select an option for each question. You may also record the time in seconds and a short reason.\n")
form_md.append("> **C1/C2/C3** are the proposed judgments. Choose **Insufficient Evidence** if the record is not enough, **None of the Above** if the options do not fit, or **Skip** if you cannot answer.\n")
form_md.append("> Use the supplied record and your clinical judgment. Some measurements, such as creatinine or eGFR, may be missing.\n")
form_md.append("\n---\n")

key_rows = []
form_csv_rows = []

last_subj = None
for i, r in enumerate(rows, 1):
    subj = r.get("subject_id", "")
    if subj != last_subj:
        form_md.append(f"\n## Patient {subj}  (task {r.get('source_record_id','')})\n")
        last_subj = subj
    q = r.get("question", "")
    qt = r.get("question_type", "")
    c1 = r.get("option_c1", ""); c2 = r.get("option_c2", ""); c3 = r.get("option_c3", "")
    form_md.append(f"\n### Question {i}  [{qt}]\n")
    form_md.append(f"**Question**: {q}\n")
    form_md.append("\n**Evidence**:\n")
    form_md.extend([l + "\n" for l in ev_lines(r.get("evidence_spans_json", "[]"))])
    form_md.append("\n**Options**:\n")
    form_md.append(f"- [ ] **C1**: {c1}\n")
    form_md.append(f"- [ ] **C2**: {c2}\n")
    form_md.append(f"- [ ] **C3**: {c3}\n")
    form_md.append(f"- [ ] **Insufficient Evidence**\n")
    form_md.append(f"- [ ] **None of the Above**\n")
    form_md.append(f"- [ ] **Skip**\n")
    form_md.append(f"\n**Your answer**: ______    **Time in seconds**: ______    **Reason (optional)**: ______________________\n")

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
                "Insufficient Evidence can be accepted when the required field is missing.",
    })

open(os.path.join(HERE, "DOCTOR_FORM_BLANK.md"), "w", encoding="utf-8").write("".join(form_md))

with open(os.path.join(HERE, "DOCTOR_FORM_BLANK.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(form_csv_rows[0].keys()))
    w.writeheader(); w.writerows(form_csv_rows)

with open(os.path.join(HERE, "ANSWER_KEY.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
    w.writeheader(); w.writerows(key_rows)

# answer key md (compact)
km = ["# HumanChoice pilot answer key — keep separate from the participant form\n",
      f"> {len(key_rows)} questions. The gold_label field is the original record label.\n",
      "> Recorded label meanings: A = ESKD/dialysis, B = CKD without ESKD, C = new AKI.\n",
      "> Option mapping: C1 supports A, C2 supports B, C3 supports C. Insufficient Evidence is also accepted when the required field is missing.\n\n",
      "| Question | Patient | Task | Type | Reference | Meaning | Selection reason |\n",
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
