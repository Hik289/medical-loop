#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anchor1_step2.py — DEBUG-anchor1 Step2 (REDUCE+COMPARE 公平 good-faith prompt)

目的: 验证 anchor_1 崩溃是否源于 prompt 欠规格 (缺 A/B/C 三类定义 + 临床鉴别 + 字母序偏置)。
good-faith = 给全三类定义 + ESKD/CKD/AKI 临床鉴别要点 + (可选) dev few-shot。
代表有临床常识的 agent 在同样 32 字段下的合理表现, 不堆 trick (不告诉先验分布)。

反 p-hacking 硬约束:
  - prompt 只在 DEV (627 train 按患者分层分出) 上调
  - few-shot 只从 train_rest (train 除 dev) 取, 不碰 dev 评估样本, 不碰 test
  - 冻结 prompt_hash 后 270 test 只跑一次, 报一个数, 禁止 test 上迭代

用法:
  --split dev|test   评估哪个集
  --fewshot N        每类 few-shot 数 (0=无)
  --promptver TAG    prompt 版本标签 (落盘区分)
"""
from __future__ import annotations
import json, re, os, sys, time, argparse, hashlib, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from runtime import data_dir, runs_dir, model_name, make_record_client
import eval_metrics as EM

JST = timezone(timedelta(hours=9))
DATA = str(data_dir())
BASE = str(runs_dir() / 'anchor1_step2')
MODEL = model_name()

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

# ---- good-faith system prompt (临床常识, 非 trick; 不含先验分布) ----
SYSTEM = (
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

def build_dev_split(seed=42, dev_per_class=None):
    """从 train 627 按患者分层分出 dev + train_rest (few-shot 来源)。确定性。"""
    if dev_per_class is None:
        dev_per_class = {"A": 92, "B": 39, "C": 20}   # ~ 24% 分层, 保持 60/26/13 比例, 共 151
    rows = load("train")
    by = {"A": [], "B": [], "C": []}
    for i, r in enumerate(rows):
        by[LABEL_FROM_ANSWER[r["answer"]]].append((i, r))
    rng = random.Random(seed)
    dev, rest = [], []
    for c in ["A", "B", "C"]:
        pool = by[c][:]
        rng.shuffle(pool)
        k = dev_per_class[c]
        dev += pool[:k]; rest += pool[k:]
    return dev, rest   # list of (orig_idx, row)

def make_fewshot(rest, n_per_class, seed=42):
    """从 train_rest 每类取 n 个 few-shot (不碰 dev/test)。"""
    if n_per_class <= 0:
        return []
    by = {"A": [], "B": [], "C": []}
    for idx, r in rest:
        by[LABEL_FROM_ANSWER[r["answer"]]].append(r)
    rng = random.Random(seed + 7)
    shots = []
    for c in ["A", "B", "C"]:
        pool = by[c][:]; rng.shuffle(pool)
        for r in pool[:n_per_class]:
            shots.append((r, c))
    rng.shuffle(shots)
    return shots

def build_messages(fields, fewshot):
    msgs = [{"role": "system", "content": SYSTEM}]
    for r, c in fewshot:
        ff = parse_fields(r["question"])
        msgs.append({"role": "user", "content": "示例患者可见字段:\n" + fields_block(ff) + "\n\n请分类 (JSON)。"})
        msgs.append({"role": "assistant", "content": json.dumps({"label": c, "evidence_ids": [], "rationale": "示例", "uncertainty": 0.2}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": "该患者可见字段如下:\n" + fields_block(fields) + "\n\n请给出三分类判断 (JSON)。"})
    return msgs

def extract_label(text):
    if not text: return None, {}
    try:
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label","")).strip().upper()
            if lab in ("A","B","C"): return lab, obj
    except Exception: pass
    m = re.search(r'\b([ABC])\b', text.upper())
    return (m.group(1), {}) if m else (None, {})

def prompt_hash(fewshot):
    h = hashlib.sha256()
    h.update(SYSTEM.encode())
    for r, c in fewshot:
        h.update(c.encode()); h.update(r["question"].encode())
    return h.hexdigest()[:16]

def call_one(client, idx, row, fields, gold, fewshot, effort, rundir):
    fpath = f"{rundir}/tasks/{idx:04d}.json"
    if os.path.exists(fpath):
        try: return idx, json.load(open(fpath, encoding="utf-8"))
        except Exception: pass
    t0 = time.time(); err=None; content=None; usage=None; pred=None; obj={}
    for attempt in range(3):
        try:
            kwargs = dict(model=MODEL, messages=build_messages(fields, fewshot), timeout=180)
            if effort: kwargs["reasoning_effort"] = effort
            r = client.chat.completions.create(**kwargs)
            content = r.choices[0].message.content
            usage = r.usage.model_dump() if r.usage else None
            pred, obj = extract_label(content)
            if pred: break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"; time.sleep(2*(attempt+1))
    rec = {"idx":idx,"gold":gold,"pred":pred,"rationale":obj.get("rationale"),
           "uncertainty":obj.get("uncertainty"),"raw":(content or "")[:400],
           "error":err if not pred else None,"latency_s":round(time.time()-t0,2),"usage":usage,
           "subject_id":fields.get("subject_id")}
    with open(fpath,"w",encoding="utf-8") as f: json.dump(rec,f,ensure_ascii=False,indent=2)
    return idx, rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev","test"], required=True)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--effort", type=str, default="high")
    ap.add_argument("--promptver", type=str, default="v1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dev, rest = build_dev_split()
    fewshot = make_fewshot(rest, args.fewshot)
    ph = prompt_hash(fewshot)

    if args.split == "dev":
        items = [(i, r) for i, (oi, r) in enumerate(dev)]
    else:
        test = load("test")
        items = list(enumerate(test))

    rundir = f"{BASE}/{args.split}_{args.promptver}_fs{args.fewshot}_{args.effort}"
    os.makedirs(f"{rundir}/tasks", exist_ok=True)
    os.makedirs(f"{rundir}/logs", exist_ok=True)
    client = make_record_client()

    print(f"[{datetime.now(JST).isoformat()}] Step2 start split={args.split} n={len(items)} "
          f"fewshot={args.fewshot} effort={args.effort} promptver={args.promptver} prompt_hash={ph} rundir={rundir}")

    records=[None]*len(items); done=0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={}
        for i,(oi,row) in enumerate(items):
            fields=parse_fields(row["question"]); gold=LABEL_FROM_ANSWER[row["answer"]]
            futs[ex.submit(call_one,client,i,row,fields,gold,fewshot,args.effort,rundir)]=i
        for fut in as_completed(futs):
            idx,rec=fut.result(); records[idx]=rec; done+=1
            if done%25==0 or done==len(items):
                print(f"  {done}/{len(items)} (idx={idx} pred={rec['pred']} gold={rec['gold']})",flush=True)

    preds=[r["pred"] if r["pred"] else "A" for r in records]
    golds=[r["gold"] for r in records]
    subs=[r["subject_id"] for r in records]
    acc=EM.record_accuracy(preds,golds)
    correct=[1.0 if p==g else 0.0 for p,g in zip(preds,golds)]
    acc_ci=EM.cluster_bootstrap_ci(correct,subs,n_boot=10000,seed=42)
    import collections
    pred_dist=dict(collections.Counter(preds))
    cm=collections.Counter((r["gold"],r["pred"] or "NA") for r in records)
    confusion={g:{p:cm[(g,p)] for p in ["A","B","C","NA"] if cm[(g,p)]} for g in ["A","B","C"]}
    n_parse_fail=sum(1 for r in records if not r["pred"])
    total_tokens=sum((r["usage"]["total_tokens"] if r.get("usage") else 0) for r in records)

    result={"project_id":"humanchoice","experiment_id":f"anchor1_step2_{args.split}",
            "timestamp":datetime.now(JST).isoformat(),
            "method":f"good-faith prompt (3类定义+临床鉴别+fewshot{args.fewshot}), gpt-5.6-sol effort={args.effort}",
            "split":args.split,"prompt_hash":ph,"promptver":args.promptver,
            "config":{"model":MODEL,"n":len(items),"fewshot_per_class":args.fewshot,"effort":args.effort,
                      "visible_fields":len(AGENT_VISIBLE)},
            "results":{"overall_acc":round(acc["overall_acc"]*100,2),
                       "overall_acc_ci95":[round(acc_ci["ci_lo"]*100,2),round(acc_ci["ci_hi"]*100,2)],
                       "macro_f1":round(acc["macro_f1"]*100,2),
                       "per_class_recall":{k:(round(v*100,1) if v is not None else None) for k,v in acc["per_class_recall"].items()},
                       "aki_recall":round(acc["aki_recall"]*100,1),"support":acc["support"],
                       "pred_dist":pred_dist,"confusion":confusion,"n_parse_fail":n_parse_fail},
            "cost":{"total_tokens":total_tokens},
            "reference":{"clean_logreg_acc":71.48,"majority_acc":60.74,"random_acc":33.3,"range_pass":[55.0,80.0]}}
    with open(f"{rundir}/result.json","w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print("\n"+"="*60)
    print(json.dumps(result["results"],ensure_ascii=False,indent=2))
    print(f"saved {rundir}/result.json")

if __name__=="__main__":
    main()
