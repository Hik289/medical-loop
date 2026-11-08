#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare classifiers trained with and without later-outcome fields.

Use the difference to measure how those fields affect prediction scores.
Also report a supervised reference using only the agent's input fields.
"""
import json, re, os
from collections import Counter
import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import data_dir, runs_dir

DATA=str(data_dir())
def parse(q):
    i=q.find("admittime"); b=q[i:] if i>=0 else q
    d={}
    for p in re.split(r'[；;]',b):
        p=p.strip()
        if ':' in p:
            k,v=p.split(':',1); d[k.strip()]=v.strip()
    return d
def miss(v): return v is None or str(v).strip().upper() in ("","NA")
def num(v):
    if miss(v): return np.nan
    try: return float(v)
    except: return np.nan

LABEL={'晚期CKD/ESKD（重度肾衰/透析相关）':0,'CKD（慢性肾病）':1,'AKI（急性肾损伤）':2}

OUTCOME=["dischtime","discharge_location","los_hospital","hospital_expire_flag",
         "icu_expire_flag","follow_up_years","survival_days","mortality_28",
         "mortality_30","mortality_90","mortality_365","los_icu","careunit",
         "stay_id","hospstay_seq","sofa"]
NUM_ALL=["los_hospital","admission_age","sofa","glucose_first","hemoglobin_a1c_first",
         "sodium_first","potassium_first","chloride_first","bicarbonate_first",
         "triglycerides_first","total_cholesterol_first","los_icu","hospstay_seq",
         "follow_up_years","survival_days","height","weight"]
BIN=["hospital_expire_flag","icu_expire_flag","mortality_28","mortality_30",
     "mortality_90","mortality_365","smoker","alcohol_abuse","diabetes",
     "myocardial_infarct","hypertension","congestive_heart_failure",
     "cerebrovascular_disease","peripheral_vascular_disease","custom_acei",
     "custom_arb","custom_Statins","custom_Metformin","custom_胰岛素"]
CAT=["gender","race","insurance","marital_status","language","admission_location",
     "discharge_location","admission_type","careunit"]

def load(s):
    rows=[json.loads(l) for l in open(f"{DATA}/{s}.jsonl",encoding="utf-8")]
    X=[parse(r["question"]) for r in rows]
    y=np.array([LABEL[r["answer"]] for r in rows])
    return X,y
Xtr,ytr=load("train"); Xte,yte=load("test")

def build(Xlist, feats_num, feats_bin, feats_cat, cat_vocab=None):
    import numpy as np
    cols=[]
    # numeric+binary: value + missing indicator
    for f in feats_num+feats_bin:
        v=np.array([num(d.get(f)) for d in Xlist])
        m=np.isnan(v).astype(float)
        vv=np.where(np.isnan(v),0.0,v)
        cols.append(vv); cols.append(m)
    # categorical one-hot
    vocab=cat_vocab if cat_vocab is not None else {}
    for f in feats_cat:
        if cat_vocab is None:
            vals=sorted({str(d.get(f)) for d in Xlist if not miss(d.get(f))})
            vocab[f]=vals
        for val in vocab[f]:
            cols.append(np.array([1.0 if str(d.get(f))==val else 0.0 for d in Xlist]))
    return np.vstack(cols).T, vocab

def run(name, fn, fb, fc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    Xtr_m,vocab=build(Xtr,fn,fb,fc)
    Xte_m,_=build(Xte,fn,fb,fc,cat_vocab=vocab)
    sc=StandardScaler().fit(Xtr_m)
    clf=LogisticRegression(max_iter=2000,class_weight=None,C=1.0)
    clf.fit(sc.transform(Xtr_m),ytr)
    pred=clf.predict(sc.transform(Xte_m))
    acc=accuracy_score(yte,pred)*100
    mf1=f1_score(yte,pred,average='macro')*100
    rec=recall_score(yte,pred,average=None)
    cm=confusion_matrix(yte,pred)
    print(f"[{name}] acc={acc:.2f} macroF1={mf1:.2f} recall A/B/C={rec[0]*100:.1f}/{rec[1]*100:.1f}/{rec[2]*100:.1f} (AKI=C)")
    return {"name":name,"acc":round(acc,2),"macro_f1":round(mf1,2),
            "recall_A_ESKD":round(rec[0]*100,1),"recall_B_CKD":round(rec[1]*100,1),
            "recall_C_AKI":round(rec[2]*100,1),"n_features":Xtr_m.shape[1],
            "confusion":cm.tolist()}

print("Majority-class baseline acc = %.2f%%"%(Counter(yte).most_common(1)[0][1]/len(yte)*100))

res=[]
# 1. FULL (original protocol, includes leakage)
res.append(run("FULL(incl leakage)",NUM_ALL,BIN,CAT))
# 2. OUTCOME-ONLY (leakage fields only)
onum=[f for f in NUM_ALL if f in OUTCOME]; obin=[f for f in BIN if f in OUTCOME]; ocat=[f for f in CAT if f in OUTCOME]
res.append(run("OUTCOME-ONLY(leakage)",onum,obin,ocat))
# 3. CLEAN visible-only (exclude all outcome/leakage + ids)
cnum=[f for f in NUM_ALL if f not in OUTCOME]; cbin=[f for f in BIN if f not in OUTCOME]; ccat=[f for f in CAT if f not in OUTCOME]
res.append(run("CLEAN(visible-only)",cnum,cbin,ccat))

json.dump({"majority_acc":round(Counter(yte).most_common(1)[0][1]/len(yte)*100,2),
           "runs":res},
          open(f"{DATA}/eda_out/leak_probe.json","w"),ensure_ascii=False,indent=2)
print("saved leak_probe.json")
