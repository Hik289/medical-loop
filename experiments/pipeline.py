#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the small-batch HumanChoice workflow with model-generated replies.

Load and split the records, run the baseline, generate questions, simulate
replies, revise actions, score results, extract features, and train A, Q, G.
The final stage repeats training on the same labels and checks the patient
split; it is a workflow check, not the full Expert Iteration experiment.
Save intermediate files locally and reuse them when resuming.
"""
from __future__ import annotations
import json, re, os, sys, time, argparse, hashlib, random, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
import numpy as np
from runtime import data_dir, runs_dir, model_name, make_record_client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_metrics as EM

JST = timezone(timedelta(hours=9))
DATA = str(data_dir())
RUNS = str(runs_dir())
MODEL = model_name()
TRAIN_PY = sys.executable

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
OPTION_TEXT = {"A":"晚期CKD/ESKD（重度肾衰/透析相关）","B":"CKD（慢性肾病）","C":"AKI（急性肾损伤）"}
QTYPES = ["Scope","Rule","Evidence","Temporal","Preference"]

# ---- frozen good-faith Base Agent prompt (prompt_hash 7b3032b2, from anchor1_step2) ----
SYSTEM_BASE = (
    "你是一名有临床常识的肾脏科医生, 正在做糖尿病相关肾脏并发症的回顾性病历分类。"
    "你将看到一位患者入院时可知的结构化字段。请在以下三类中判断最可能的分类。\n\n"
    "【三类定义与鉴别要点】\n"
    "A = 晚期CKD/ESKD（重度肾衰/透析相关）: 慢性肾病的终末期。典型线索——高龄; 长期糖尿病合并高血压; "
    "多系统慢性心血管共病负担重(充血性心衰/心梗/脑血管病/外周血管病常并存); 长期肾衰的代偿紊乱"
    "(碳酸氢盐偏低的慢性代谢性酸中毒、钾偏高的慢性高钾倾向); 常已在用 ACEI/ARB/他汀等慢病长期药物。\n"
    "B = CKD（慢性肾病, 非终末期）: 慢性肾功能受损但未达 ESKD。典型线索——有糖尿病/高血压等慢性基础; "
    "肾功能异常存在但共病负担与电解质紊乱程度轻于 ESKD; 处于慢性稳定或缓慢进展阶段。\n"
    "C = AKI（急性肾损伤）: 急性起病的肾损伤, 多可逆。典型线索——急诊/观察类入院提示急性事件; "
    "相对年轻、慢性共病较少; 电解质为急性紊乱而非长期代偿模式; 缺乏长期终末期慢病积累的证据。\n\n"
    "【重要约束】本数据不含血清肌酐/eGFR/UACR 及其轨迹, 不得臆造这些数值。"
    "请基于人口学、慢性共病负担、长期用药、入院类型与首次电解质等间接证据, 综合判断最符合哪一类的整体病程阶段。"
    "严重度大致为 ESKD(A) > CKD(B) ; AKI(C) 是独立的急性过程, 不是慢性严重度的延伸。\n\n"
    "严格输出 JSON: {\"label\": \"A|B|C\", \"evidence_ids\": [用到的字段名], \"rationale\": \"一句话依据\", "
    "\"uncertainty\": 0.0-1.0}。不要输出任何额外文字。"
)

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

def fields_block(fields):
    return "\n".join(f"{k}: {fields.get(k,'NA')}" for k in AGENT_VISIBLE)

def miss(v): return v is None or str(v).strip().upper() in ("", "NA")

def extract_json(text):
    if not text: return {}
    try:
        m = re.search(r'\{.*\}', text, re.S)
        if m: return json.loads(m.group(0))
    except Exception: pass
    return {}

def llm_call(client, messages, tag=""):
    """Call the model with retries and return its text and usage counts."""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages, reasoning_effort="high", timeout=180)
            return r.choices[0].message.content, (r.usage.model_dump() if r.usage else None)
        except Exception as e:
            if attempt == 2:
                return None, {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            time.sleep(2*(attempt+1))
    return None, None

# =====================================================================
# S0 — data load + split freeze + mini-batch select
# =====================================================================
def s0_load(run_dir, n_batch, seed=42):
    tr = load("train")
    smap = json.load(open(f"{RUNS}/split_map.json", encoding="utf-8"))
    # deterministic 3-way patient-level split (D_A/D_Q/D_G) within train (anchor_2b)
    def assign_group(subj):
        h = int(hashlib.sha256(str(subj).encode()).hexdigest(), 16)
        return ["D_A","D_Q","D_G"][h % 3]
    # build mini-batch: stratified by label, 20 patients
    by = defaultdict(list)
    for i, r in enumerate(tr):
        by[LABEL_FROM_ANSWER[r["answer"]]].append((i, r))
    rng = random.Random(seed)
    per = {"A": int(round(n_batch*0.55)), "B": int(round(n_batch*0.25)), "C": n_batch}
    # take proportional; ensure sum==n_batch, AKI(C) guaranteed representation
    quota = {"A": max(1,int(round(n_batch*0.5))), "B": max(1,int(round(n_batch*0.25))),
             "C": max(1,n_batch - int(round(n_batch*0.5)) - int(round(n_batch*0.25)))}
    batch = []
    for c in ["A","B","C"]:
        pool = by[c][:]; rng.shuffle(pool)
        batch += pool[:quota[c]]
    rng.shuffle(batch)
    tasks = []
    for oi, row in batch:
        f = parse_fields(row["question"])
        subj = str(f.get("subject_id"))
        tasks.append({
            "task_id": f"{oi:04d}",
            "subject_id": subj,
            "hadm_id": str(f.get("hadm_id")),
            "split": smap.get(subj, "unknown"),
            "group": assign_group(subj),
            "fields": {k: f.get(k,"NA") for k in AGENT_VISIBLE},
            "gold": LABEL_FROM_ANSWER[row["answer"]],
        })
    os.makedirs(run_dir, exist_ok=True)
    # split freeze contract
    with open(f"{run_dir}/batch_tasks.json","w",encoding="utf-8") as fp:
        json.dump(tasks, fp, ensure_ascii=False, indent=2)
    # anchor_3 recheck on batch: no leakage fields in agent_visible, all in train split
    leak = set(["answer","answer_idx","options","meta_info","hospital_expire_flag","dischtime",
                "discharge_location","los_hospital","sofa","stay_id","careunit","hospstay_seq",
                "subject_id","hadm_id"])
    a3_ok = all(len(set(t["fields"].keys()) & leak)==0 for t in tasks)
    a3_split = all(t["split"]=="train" for t in tasks)
    assertions = {"anchor_3_no_leak_in_visible": a3_ok, "anchor_3_all_train_split": a3_split,
                  "n_tasks": len(tasks),
                  "group_dist": dict(Counter(t["group"] for t in tasks)),
                  "label_dist": dict(Counter(t["gold"] for t in tasks))}
    json.dump(assertions, open(f"{run_dir}/s0_assertions.json","w",encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if not a3_ok or not a3_split:
        raise RuntimeError("Record visibility or frozen train split invariant failed")
    return tasks, assertions

# =====================================================================
# S1 — Autonomous Agent
# =====================================================================
def s1_autonomous(client, t, run_dir):
    fp = f"{run_dir}/autonomous/{t['task_id']}.json"
    if os.path.exists(fp):
        return json.load(open(fp, encoding="utf-8"))
    msgs = [{"role":"system","content":SYSTEM_BASE},
            {"role":"user","content":"该患者可见字段如下:\n"+fields_block(t["fields"])+"\n\n请给出三分类判断 (JSON)。"}]
    t0=time.time(); content,usage=llm_call(client,msgs,"s1")
    obj=extract_json(content)
    lab=str(obj.get("label","")).strip().upper()
    if lab not in ("A","B","C"): lab="A"
    rec={"task_id":t["task_id"],"a0":lab,"evidence_ids":obj.get("evidence_ids",[]),
         "uncertainty":obj.get("uncertainty"),"rationale":obj.get("rationale"),
         "gold":t["gold"],"model_version":MODEL,"prompt_hash":"7b3032b2",
         "tokens":(usage or {}).get("total_tokens"),"latency":round(time.time()-t0,2),
         "raw":(content or "")[:400]}
    os.makedirs(os.path.dirname(fp),exist_ok=True)
    json.dump(rec,open(fp,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    return rec

# =====================================================================
# S2 — candidate question generation (<=3 runtime questions)
# =====================================================================
S2_SYS = (
    "你是一个临床决策澄清助手。给定一位患者的可见字段和 AI 初步分类, 生成最多3个"
    "能帮助消歧三分类(A=晚期CKD/ESKD, B=CKD, C=AKI)的局部选择题。每题须: "
    "(1) 属于五型之一 Scope/Rule/Evidence/Temporal/Preference; "
    "(2) 仅引用已给的 agent_visible 字段作证据, 不得臆造肌酐/eGFR/UACR 等不存在的检验; "
    "(3) 2-3 个实质选项 + 固定附加 Insufficient/None/Skip; "
    "(4) 指出该题影响哪个行动字段 (affected_action_field, 如 'label')。\n"
    "严格输出 JSON: {\"questions\":[{\"question_id\":\"q1\",\"type\":\"...\",\"question\":\"...\","
    "\"evidence\":[字段名],\"substantive_options\":[\"...\",\"...\"],"
    "\"other_options\":[\"Insufficient\",\"None\",\"Skip\"],\"affected_action_field\":\"label\"}]}。"
)
def s2_questions(client, t, a0, run_dir):
    fp=f"{run_dir}/questions/{t['task_id']}.json"
    if os.path.exists(fp): return json.load(open(fp,encoding="utf-8"))
    msgs=[{"role":"system","content":S2_SYS},
          {"role":"user","content":f"患者可见字段:\n{fields_block(t['fields'])}\n\nAI初步分类: {a0} ({OPTION_TEXT[a0]})\n\n请生成最多3个消歧局部选择题 (JSON)。"}]
    content,usage=llm_call(client,msgs,"s2")
    obj=extract_json(content)
    qs=obj.get("questions",[]) if isinstance(obj,dict) else []
    # program validation: cap 3, ensure fields, ensure other_options
    clean=[]
    for i,q in enumerate(qs[:3]):
        if not isinstance(q,dict): continue
        qt=q.get("type","Scope"); qt=qt if qt in QTYPES else "Scope"
        subs=q.get("substantive_options",[]) or []
        if len(subs)<2: continue
        ev=[e for e in (q.get("evidence",[]) or []) if e in AGENT_VISIBLE]
        clean.append({"question_id":f"q{i+1}","type":qt,"question":q.get("question","")[:300],
                      "evidence":ev,"substantive_options":subs[:3],
                      "other_options":["Insufficient","None","Skip"],
                      "affected_action_field":q.get("affected_action_field","label")})
    rec={"task_id":t["task_id"],"checkpoint_idx":0,"Q_pool":clean,
         "prompt_hash":"s2_v1","tokens":(usage or {}).get("total_tokens")}
    os.makedirs(os.path.dirname(fp),exist_ok=True)
    json.dump(rec,open(fp,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    return rec

# =====================================================================
# S3 — simulated-user answers (gpt-5.6-sol as doctor, only agent_visible)
# =====================================================================
S3_SYS = (
    "你是一名真实的肾脏科医生, 正在回答一道关于某患者的局部临床选择题。"
    "你只能看到题目给出的 agent_visible 证据, 看不到标准答案或参考诊断。"
    "请基于你的临床判断选一个最合适的选项。若证据确实不足以判断, 选 Insufficient; "
    "若都不对选 None; 若无法回答选 Skip。严格输出 JSON: {\"choice\":\"选项原文\",\"reason\":\"一句话\"}。"
)
def s3_simuser(client, t, qrec, run_dir):
    """Simulate a reply to the first candidate question using the same evidence."""
    fp=f"{run_dir}/sim_answers/{t['task_id']}.json"
    if os.path.exists(fp): return json.load(open(fp,encoding="utf-8"))
    qpool=qrec["Q_pool"]
    if not qpool:
        rec={"task_id":t["task_id"],"q_t_id":None,"r_t":"no_response","r_t_type":"no_response",
             "simulated_time":0,"n_participants":0,"participant_responses":[]}
        os.makedirs(os.path.dirname(fp),exist_ok=True)
        json.dump(rec,open(fp,"w",encoding="utf-8"),ensure_ascii=False,indent=2); return rec
    q=qpool[0]  # q_t = first question (runtime selects 1)
    ev_block="\n".join(f"{e}: {t['fields'].get(e,'NA')}" for e in q["evidence"]) or "(题目未指定具体证据字段, 可参考全部可见字段)"
    allopts=q["substantive_options"]+q["other_options"]
    opt_txt="\n".join(f"- {o}" for o in allopts)
    # simulate n_participants=2 (requirement (c): n>=2 for comparison)
    responses=[]
    toks=0
    for pid in range(2):
        msgs=[{"role":"system","content":S3_SYS},
              {"role":"user","content":f"题型: {q['type']}\n题目: {q['question']}\n\n相关证据:\n{ev_block}\n\n选项:\n{opt_txt}\n\n请作答 (JSON)。"}]
        content,usage=llm_call(client,msgs,"s3")
        toks+=(usage or {}).get("total_tokens") or 0
        obj=extract_json(content)
        choice=str(obj.get("choice","")).strip()
        # map to canonical
        matched=None
        for o in allopts:
            if o and (o in choice or choice in o): matched=o; break
        if matched is None: matched=choice[:60] if choice else "Insufficient"
        responses.append({"participant_id":pid,"choice":matched,"reason":obj.get("reason","")[:120]})
    # majority r_t
    cc=Counter(r["choice"] for r in responses)
    r_t=cc.most_common(1)[0][0]
    low=r_t.lower()
    if "insufficient" in low: r_type="insufficient"
    elif low=="none" or "none" in low: r_type="none"
    elif "skip" in low: r_type="skip"
    else: r_type="substantive"
    # simulated_time by qtype (placeholder, NOT real human minutes)
    tmap={"Scope":45,"Rule":60,"Evidence":75,"Temporal":50,"Preference":40}
    rec={"task_id":t["task_id"],"q_t_id":q["question_id"],"q_t_type":q["type"],
         "r_t":r_t,"r_t_type":r_type,"simulated_time":tmap.get(q["type"],60),
         "n_participants":2,"participant_responses":responses,
         "evidence_used":q["evidence"],"tokens":toks,"note":"simulated_user (pipeline-only, not formal result)"}
    os.makedirs(os.path.dirname(fp),exist_ok=True)
    json.dump(rec,open(fp,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    return rec

# =====================================================================
# S4 — action revision a1 = f(x, q, r)  (+ hard constraint check)
# =====================================================================
S4_SYS = (
    "你是临床记录审核 Agent。你之前给出了一个三分类初判。现在一位医生对一道局部选择题"
    "做了回答。请结合该回答重新审视你的分类, 给出修改后的分类 a1 (A/B/C)。"
    "硬约束: 不得臆造不存在的检验值; 若医生选 None/Insufficient/Skip 不要据此编造新结论。"
    "严格输出 JSON: {\"label\":\"A|B|C\",\"changed\":true/false,\"rationale\":\"一句话\"}。"
)
def s4_revise(client, t, a0, qrec, srec, run_dir):
    fp=f"{run_dir}/interactions/{t['task_id']}_0.json"
    if os.path.exists(fp): return json.load(open(fp,encoding="utf-8"))
    r_type=srec["r_t_type"]; branch="ask"
    constraint_ok=True; constraint_note=""
    if r_type in ("skip","no_response"):
        # Skip/NoResponse -> fallback to autonomous (a1=a0)
        a1=a0; changed=False; branch="wait"; constraint_note="skip/no_response->autonomous fallback"
        toks=0
    elif r_type in ("none","insufficient"):
        # None/Insufficient -> no fabrication; keep a0 (A not allowed to invent)
        a1=a0; changed=False; constraint_note="none/insufficient->no fabrication, keep a0"
        toks=0
    else:
        q=qrec["Q_pool"][0]
        msgs=[{"role":"system","content":S4_SYS},
              {"role":"user","content":f"患者可见字段:\n{fields_block(t['fields'])}\n\n你的初判: {a0}\n\n局部题: {q['question']}\n医生回答: {srec['r_t']}\n\n请给出修改后分类 (JSON)。"}]
        content,usage=llm_call(client,msgs,"s4")
        toks=(usage or {}).get("total_tokens") or 0
        obj=extract_json(content)
        lab=str(obj.get("label","")).strip().upper()
        a1=lab if lab in ("A","B","C") else a0
        changed=(a1!=a0)
    rec={"task_id":t["task_id"],"checkpoint_idx":0,"q_t_id":srec["q_t_id"],
         "r_t":srec["r_t"],"r_t_type":r_type,"human_seconds":srec["simulated_time"],
         "simulated_time":srec["simulated_time"],
         "n_participants":srec["n_participants"],"participant_responses":srec["participant_responses"],
         "a0":a0,"a1":a1,"A_t":[a0,a1],"changed":changed,
         "constraint_check":{"ok":constraint_ok,"note":constraint_note},
         "policy_version":"pi_0","branch":branch,"model_version":MODEL,
         "prompt_hash":"s4_v1","tokens":toks,
         "note":"simulated (pipeline-only)"}
    os.makedirs(os.path.dirname(fp),exist_ok=True)
    json.dump(rec,open(fp,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    return rec

# =====================================================================
# S5 — final scoring Y / U  (simulated scorer, requirement(b) Unresolved rubric)
# =====================================================================
def s5_score(t, irec, run_dir):
    """
    Score the baseline and revised record labels against the stored label.

    Save both branches. The review rule marks specific claims as unsupported
    when the required kidney measurements are absent. The cost settings are
    fixed placeholders for this workflow check.
    """
    gold=t["gold"]
    lam_H=0.0005; lam_C=0.0  # placeholder, frozen before test in real exp
    out=[]
    for branch,action in [("wait",irec["a0"]),("ask",irec["a1"])]:
        Y=1.0 if action==gold else 0.0
        # requirement(b): evidence boundary -> allow_unresolved True for CKD/AKI/ESKD confirm
        # scorer rubric: a concrete label under absent gold-standard = unsupported inference
        # (simulated rubric; real scorer by review doctors). We record both correctness Y and unsupported flag.
        unsupported=1  # concrete claim under absent creatinine/eGFR/UACR
        T_H=irec["simulated_time"] if branch=="ask" else 0
        C_API=0
        U=Y - lam_H*T_H - lam_C*C_API
        fp=f"{run_dir}/outcomes/{t['task_id']}_0_{branch}_{action}.json"
        rec={"task_id":t["task_id"],"checkpoint":0,"branch":branch,"action":action,
             "Y":Y,"U_components":{"Y":Y,"T_H":T_H,"C_API":C_API,"lambda_H":lam_H,"lambda_C":lam_C},
             "U":round(U,4),"unsupported":unsupported,
             "scorer":"simulated","rubric_version":"b_unresolved_v1"}
        os.makedirs(os.path.dirname(fp),exist_ok=True)
        json.dump(rec,open(fp,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        out.append(rec)
    return out

# =====================================================================
# S6 — G/Q/A feature extraction (<=64 dim, deterministic)
# =====================================================================
def _num(v,default=0.0):
    try: return float(v)
    except Exception: return default

def s6_features(t, a0rec, qrec, irec, outs, feat_schema):
    """Build fixed numerical features for G, Q, and A and pad them to length 64."""
    f=t["fields"]
    n_missing=sum(1 for k in AGENT_VISIBLE if miss(f.get(k)))
    qpool=qrec["Q_pool"]; q=qpool[0] if qpool else None
    # --- G features (when-to-ask): checkpoint, evidence state, budget, action divergence ---
    g=[
        0.0,                                   # checkpoint_idx
        float(len(qpool)),                     # n candidate questions
        float(n_missing)/len(AGENT_VISIBLE),   # missing ratio
        1.0 if q else 0.0,                     # has question
        float(QTYPES.index(q["type"]))/4 if q else 0.0,  # q type ordinal
        float(len(q["question"]))/300 if q else 0.0,     # q length
        float(len(q["evidence"]))/5 if q else 0.0,       # evidence refs
        1.0 if (irec["a0"]!=irec["a1"]) else 0.0,        # action divergence
        _num(a0rec.get("uncertainty"),0.5),              # base uncertainty
        float(irec["simulated_time"])/120,               # est reading time
        1.0,                                             # remaining budget (norm)
    ]
    # --- Q features (how-to-ask): type, length, options, evidence, affects action ---
    q_feat=[
        float(QTYPES.index(q["type"]))/4 if q else 0.0,
        float(len(q["question"]))/300 if q else 0.0,
        float(len(q["substantive_options"]))/3 if q else 0.0,
        float(len(q["evidence"]))/5 if q else 0.0,
        float(n_missing)/len(AGENT_VISIBLE),
        1.0 if (irec["a0"]!=irec["a1"]) else 0.0,  # option changes downstream action
        _num(a0rec.get("uncertainty"),0.5),
        float(irec["simulated_time"])/120,
    ]
    # --- A features (revision): state, human choice, action diff, constraint, adopted ---
    a_feat=[
        float(ord(irec["a0"])-65)/2,           # a0 code
        1.0 if irec["r_t_type"]=="substantive" else 0.0,
        1.0 if irec["r_t_type"]=="insufficient" else 0.0,
        1.0 if irec["r_t_type"]=="none" else 0.0,
        1.0 if irec["r_t_type"]=="skip" else 0.0,
        1.0 if irec["changed"] else 0.0,       # choice adopted (action changed)
        1.0 if irec["constraint_check"]["ok"] else 0.0,
        _num(a0rec.get("uncertainty"),0.5),
        float(n_missing)/len(AGENT_VISIBLE),
    ]
    def pad(v): return (v+[0.0]*64)[:64]
    return {"G":pad(g),"Q":pad(q_feat),"A":pad(a_feat)}

# =====================================================================
# main orchestration
# =====================================================================
def run_pipeline(run_id, n_batch, workers, iters):
    run_dir=f"{RUNS}/{run_id}"
    os.makedirs(run_dir,exist_ok=True)
    log=lambda m: print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] {m}",flush=True)
    client=make_record_client()

    # ---- S0 ----
    log("S0: load + split freeze + mini-batch select")
    tasks,s0a=s0_load(run_dir,n_batch)
    log(f"S0 done: n={len(tasks)} groups={s0a['group_dist']} labels={s0a['label_dist']} "
        f"anchor3_noleak={s0a['anchor_3_no_leak_in_visible']} all_train={s0a['anchor_3_all_train_split']}")

    # ---- S1 (concurrent) ----
    log(f"S1: Autonomous Agent (concurrent workers={workers})")
    a0map={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(s1_autonomous,client,t,run_dir):t["task_id"] for t in tasks}
        done=0
        for fut in as_completed(futs):
            rec=fut.result(); a0map[rec["task_id"]]=rec; done+=1
            if done%5==0 or done==len(tasks): log(f"  S1 {done}/{len(tasks)}")
    acc=sum(1 for t in tasks if a0map[t["task_id"]]["a0"]==t["gold"])/len(tasks)
    log(f"S1 done: batch autonomous acc={acc:.3f} (n={len(tasks)}, mini-batch not anchor_1 judge)")

    # ---- S2 (concurrent) ----
    log("S2: candidate question generation")
    qmap={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(s2_questions,client,t,a0map[t["task_id"]]["a0"],run_dir):t["task_id"] for t in tasks}
        done=0
        for fut in as_completed(futs):
            rec=fut.result(); qmap[rec["task_id"]]=rec; done+=1
            if done%5==0 or done==len(tasks): log(f"  S2 {done}/{len(tasks)}")
    nq=[len(qmap[t['task_id']]['Q_pool']) for t in tasks]
    log(f"S2 done: avg questions/task={np.mean(nq):.2f} (max3)")

    # ---- S3 (concurrent) ----
    log("S3: simulated-user answers (n_participants=2)")
    smap_ans={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(s3_simuser,client,t,qmap[t["task_id"]],run_dir):t["task_id"] for t in tasks}
        done=0
        for fut in as_completed(futs):
            rec=fut.result(); smap_ans[rec["task_id"]]=rec; done+=1
            if done%5==0 or done==len(tasks): log(f"  S3 {done}/{len(tasks)}")
    rtypes=Counter(smap_ans[t['task_id']]['r_t_type'] for t in tasks)
    log(f"S3 done: r_t_type dist={dict(rtypes)}")

    # ---- S4 (concurrent) ----
    log("S4: action revision a1")
    imap={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(s4_revise,client,t,a0map[t["task_id"]]["a0"],qmap[t["task_id"]],smap_ans[t["task_id"]],run_dir):t["task_id"] for t in tasks}
        done=0
        for fut in as_completed(futs):
            rec=fut.result(); imap[rec["task_id"]]=rec; done+=1
            if done%5==0 or done==len(tasks): log(f"  S4 {done}/{len(tasks)}")
    n_changed=sum(1 for t in tasks if imap[t['task_id']]['changed'])
    log(f"S4 done: n_changed={n_changed}/{len(tasks)}")

    # ---- S5 ----
    log("S5: final scoring Y/U (simulated scorer, Unresolved rubric)")
    base_succ=[]; hc_succ=[]; golds=[]; subs=[]; unsup_base=[]; unsup_hc=[]
    for t in tasks:
        outs=s5_score(t,imap[t["task_id"]],run_dir)
        wait=[o for o in outs if o["branch"]=="wait"][0]
        ask=[o for o in outs if o["branch"]=="ask"][0]
        base_succ.append(bool(wait["Y"])); hc_succ.append(bool(ask["Y"]))
        unsup_base.append(wait["unsupported"]); unsup_hc.append(ask["unsupported"])
        golds.append(t["gold"]); subs.append(t["subject_id"])
    # metrics (eval_metrics): rescue/harm/net_rescue + per-class + bootstrap
    rh=EM.rescue_harm(base_succ,hc_succ)
    nr_ci=EM.net_rescue_bootstrap_ci(base_succ,hc_succ,subs,n_boot=2000,seed=42)
    rh_C=EM.rescue_harm(base_succ,hc_succ,gold=golds,stratify_class="C")
    base_acc=EM.record_accuracy([("C" if False else g) for g in golds],golds)  # placeholder
    base_pred=[ "A" for _ in tasks]  # not used
    acc_base=EM.record_accuracy([imap[t["task_id"]]["a0"] for t in tasks],golds)
    acc_hc=EM.record_accuracy([imap[t["task_id"]]["a1"] for t in tasks],golds)
    log(f"S5 done: base_acc={acc_base['overall_acc']:.3f} hc_acc={acc_hc['overall_acc']:.3f} "
        f"net_rescue={rh['net_rescue']:.3f} CI=[{nr_ci['ci_lo']:.3f},{nr_ci['ci_hi']:.3f}]")

    # ---- S6 ----
    log("S6: G/Q/A feature extraction (<=64 dim)")
    feats={"G":[],"Q":[],"A":[]}
    feat_meta=[]
    for t in tasks:
        fv=s6_features(t,a0map[t["task_id"]],qmap[t["task_id"]],imap[t["task_id"]],None,None)
        for m in ["G","Q","A"]: feats[m].append(fv[m])
        feat_meta.append({"task_id":t["task_id"],"group":t["group"],"gold":t["gold"]})
    # targets:
    # A target: a* = gold (teacher uses external Y to pick best action) -> but A learns revised correctness
    #   label: 1 if a1==gold (revision led to correct) else 0  (NLL on A branch quality)
    # Q target: pairwise -> single question so use U^Ask - U^Wait sign (did asking help)
    # G target: ask vs wait -> 1 if U_ask>U_wait (requirement(d): use actual q_t's U^Ask)
    y_A=[]; y_Q=[]; y_G=[]
    for t in tasks:
        ir=imap[t["task_id"]]
        y_A.append(1.0 if ir["a1"]==t["gold"] else 0.0)
        # U_ask and U_wait from outcomes
        ua=1.0 if ir["a1"]==t["gold"] else 0.0
        uw=1.0 if ir["a0"]==t["gold"] else 0.0
        y_Q.append(1.0 if ua>=uw else 0.0)
        y_G.append(1.0 if ua>uw else 0.0)
    os.makedirs(f"{run_dir}/features",exist_ok=True)
    for m in ["G","Q","A"]:
        X=np.array(feats[m],dtype=np.float32)
        yv={"G":y_G,"Q":y_Q,"A":y_A}[m]
        groups=[fm["group"] for fm in feat_meta]
        # module m trained on its own patient group (D_A/D_Q/D_G) - requirement(a)
        np.savez(f"{run_dir}/features/{m}_train.npz",
                 X=X,y=np.array(yv,dtype=np.float32),
                 groups=np.array(groups),task_ids=np.array([fm["task_id"] for fm in feat_meta]))
    schema={"dim":64,"modules":["G","Q","A"],"normalization":"train_only","frozen":True,
            "G_feat_count":11,"Q_feat_count":8,"A_feat_count":9}
    schema_bytes=json.dumps(schema,sort_keys=True).encode()
    schema["sha256"]=hashlib.sha256(schema_bytes).hexdigest()
    json.dump(schema,open(f"{run_dir}/features/feature_schema.json","w"),ensure_ascii=False,indent=2)
    log(f"S6 done: features saved, schema_sha={schema['sha256'][:12]}")

    # ---- S7 (torch, subprocess to the current Python environment) ----
    log("S7: three-module training (A->Q->G coordinate-wise, torch)")
    tr_out=f"{run_dir}/train_log_iter0.json"
    cmd=[TRAIN_PY, f"{os.path.dirname(os.path.abspath(__file__))}/train_modules.py",
         "--run_dir",run_dir,"--iter","0"]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
    print(p.stdout[-2000:] if p.stdout else "")
    if p.returncode!=0:
        raise RuntimeError("S7 training failed; inspect local run artifacts")
    else:
        tl=json.load(open(tr_out)) if os.path.exists(tr_out) else {}
        log(f"S7 done: total_params={tl.get('total_params')} "
            f"converged={tl.get('all_converged')} dev_nan={tl.get('any_nan')}")

    # ---- S8 — one round Expert Iteration (anchor_2b recheck) ----
    log("S8: Expert Iteration round 1 (target refresh + anchor_2b split-invariant assert)")
    # anchor_2b: refresh targets; assert patient->split unchanged, D_A/D_Q/D_G mutually exclusive
    smap=json.load(open(f"{RUNS}/split_map.json",encoding="utf-8"))
    def assign_group(subj):
        h=int(hashlib.sha256(str(subj).encode()).hexdigest(),16)
        return ["D_A","D_Q","D_G"][h%3]
    a2b_pass=True; a2b_detail=[]
    for t in tasks:
        subj=t["subject_id"]
        split_now=smap.get(subj,"unknown")
        group_now=assign_group(subj)
        if split_now!=t["split"]: a2b_pass=False; a2b_detail.append(f"{subj} split changed")
        if group_now!=t["group"]: a2b_pass=False; a2b_detail.append(f"{subj} group changed")
    # mutual exclusion of D_A/D_Q/D_G patients
    gp=defaultdict(set)
    for t in tasks: gp[t["group"]].add(t["subject_id"])
    inter=(gp["D_A"]&gp["D_Q"])|(gp["D_A"]&gp["D_G"])|(gp["D_Q"]&gp["D_G"])
    if inter: a2b_pass=False; a2b_detail.append(f"group overlap {inter}")
    log(f"S8 anchor_2b: split-invariant PASS={a2b_pass} (detail={a2b_detail[:3] if a2b_detail else 'none'})")

    # refresh: iter1 re-train with same frozen split (pipeline validation - targets identical here)
    os.makedirs(f"{run_dir}/iter1",exist_ok=True)
    cmd=[TRAIN_PY, f"{os.path.dirname(os.path.abspath(__file__))}/train_modules.py",
         "--run_dir",run_dir,"--iter","1"]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
    if p.returncode==0 and os.path.exists(f"{run_dir}/train_log_iter1.json"):
        log("S8 done: iter1 retrain complete")
        ei_ok=True
    else:
        print("S8 iter1 STDERR:",p.stderr[-1500:] if p.stderr else "")
        ei_ok=False

    # ei_curve.json
    ei_curve=[]
    for k in [0,1]:
        tl=f"{run_dir}/train_log_iter{k}.json"
        tld=json.load(open(tl)) if os.path.exists(tl) else {}
        ei_curve.append({"iter":k,"Final_Quality":round(acc_hc["overall_acc"],4),
                         "Rescue":round(rh["rescue"],4),"Harm":round(rh["harm"],4),
                         "Net_Rescue":round(rh["net_rescue"],4),
                         "train_converged":tld.get("all_converged"),
                         "n_participants_total":sum(smap_ans[t["task_id"]]["n_participants"] for t in tasks)})
    json.dump(ei_curve,open(f"{run_dir}/ei_curve.json","w"),ensure_ascii=False,indent=2)

    # ---- summary ----
    n_part_total=sum(smap_ans[t["task_id"]]["n_participants"] for t in tasks)
    train_log0=json.load(open(f"{run_dir}/train_log_iter0.json")) if os.path.exists(f"{run_dir}/train_log_iter0.json") else {}
    summary={
        "run_id":run_id,"timestamp":datetime.now(JST).isoformat(),
        "mode":"simulated_minibatch","n_tasks":len(tasks),
        "note":"Phase2 simulated-user pipeline validation. NOT formal augmentation result.",
        "S0":s0a,
        "S1_autonomous":{"batch_acc":round(acc,4),"n":len(tasks)},
        "S2_questions":{"avg_per_task":round(float(np.mean(nq)),2),"max":3},
        "S3_simuser":{"r_t_type_dist":dict(rtypes),"n_participants_each":2},
        "S4_revision":{"n_changed":n_changed},
        "S5_scoring":{
            "base_acc":round(acc_base["overall_acc"],4),"base_macro_f1":round(acc_base["macro_f1"],4),
            "base_per_class_recall":acc_base["per_class_recall"],
            "hc_acc":round(acc_hc["overall_acc"],4),"hc_macro_f1":round(acc_hc["macro_f1"],4),
            "hc_per_class_recall":acc_hc["per_class_recall"],
            "net_rescue":round(rh["net_rescue"],4),"rescue":round(rh["rescue"],4),"harm":round(rh["harm"],4),
            "net_rescue_ci95":[round(nr_ci["ci_lo"],4),round(nr_ci["ci_hi"],4)],
            "net_rescue_C_AKI":round(rh_C["net_rescue"],4) if not np.isnan(rh_C["net_rescue"]) else None,
            "unsupported_base":float(np.mean(unsup_base)),"unsupported_hc":float(np.mean(unsup_hc)),
            "scorer":"simulated","rubric":"b_unresolved_v1"},
        "S6_features":{"dim":64,"schema_sha":schema["sha256"][:16]},
        "S7_training":{"total_params":train_log0.get("total_params"),
                       "all_converged":train_log0.get("all_converged"),
                       "any_nan":train_log0.get("any_nan"),
                       "per_module":train_log0.get("per_module")},
        "S8_expert_iteration":{"anchor_2b_split_invariant_pass":a2b_pass,
                               "ei_round_complete":ei_ok,"n_participants_total":n_part_total},
        "contract_files":{
            "split_map":os.path.exists(f"{RUNS}/split_map.json"),
            "batch_tasks":os.path.exists(f"{run_dir}/batch_tasks.json"),
            "autonomous":len(os.listdir(f"{run_dir}/autonomous")) if os.path.exists(f"{run_dir}/autonomous") else 0,
            "questions":len(os.listdir(f"{run_dir}/questions")) if os.path.exists(f"{run_dir}/questions") else 0,
            "interactions":len(os.listdir(f"{run_dir}/interactions")) if os.path.exists(f"{run_dir}/interactions") else 0,
            "outcomes":len(os.listdir(f"{run_dir}/outcomes")) if os.path.exists(f"{run_dir}/outcomes") else 0,
            "features":os.path.exists(f"{run_dir}/features/feature_schema.json"),
            "ei_curve":os.path.exists(f"{run_dir}/ei_curve.json"),
        },
    }
    json.dump(summary,open(f"{run_dir}/pipeline_summary.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    log("PIPELINE COMPLETE")
    print("\n"+"="*60)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_id",default="pipeline_sim_mb20", help='Name of the local run folder')
    ap.add_argument("--n_batch",type=int,default=20, help='Number of training records in the batch')
    ap.add_argument("--workers",type=int,default=12, help='Number of parallel model calls')
    ap.add_argument("--iters",type=int,default=1, help='Legacy option; this workflow runs its fixed training check')
    a=ap.parse_args()
    run_pipeline(a.run_id,a.n_batch,a.workers,a.iters)
