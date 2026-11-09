"""Exploratory replay of approximate Ask/Wait checkpoint branches.

Branch records contain full-task evaluated returns after a forced first decision.
Future decisions follow the frozen continuation policy that collected the cache.
This is not a publication result or a matched-state causal comparison. The
source cache does not certify identical environment states across branches;
even its observed feature matrices differ for many nominal checkpoints. Use
run_replay_study.py for the reported static replay and run_online_ablation.py
for new episodes. This script is retained to make the exploratory audit clear.
"""
import argparse,glob,json,hashlib,itertools,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from replay_modules.mlp import train_bce,train_pairwise

def h(s):return hashlib.sha256(str(s).encode()).hexdigest()
def fit(records,seed,variant='full'):
    models={};log={};X=[];y=[]
    for r in records:
        if r['group']!='D_A':continue
        for q in r['questions']:
            if q['v0']!=q['v1'] and q['xa'] is not None:
                X.append(q['xa']);y.append(float(q['v1']>q['v0']))
    if len(y)>=4 and variant!='no_LA':models['A']=train_bce(np.array(X),np.array(y),seed,30)[0]
    log['A']={'n':len(y),'fitted':'A' in models}
    teacher={} if variant=='stale_labels' else models
    xp=[];xn=[]
    for r in records:
        if r['group']!='D_Q':continue
        qs=r['questions'];vals=[ask(q,teacher) for q in qs]
        for i in range(len(qs)):
            for j in range(i+1,len(qs)):
                if vals[i]==vals[j]:continue
                hi,lo=(i,j) if vals[i]>vals[j] else (j,i)
                xp.append(qs[hi]['xq']);xn.append(qs[lo]['xq'])
    if len(xp)>=4 and variant!='no_LQ':models['Q']=train_pairwise(np.array(xp),np.array(xn),seed,30)[0]
    log['Q']={'n':len(xp),'fitted':'Q' in models}
    teacher={} if variant=='stale_labels' else models
    X=[];y=[]
    for r in records:
        if r['group']!='D_G':continue
        j=question(r,teacher);X.append(r['questions'][j]['xq']);y.append(float(ask(r['questions'][j],teacher)>r['wait']))
    if len(y)>=4 and variant!='no_LG':models['G']=train_bce(np.array(X),np.array(y),seed,30)[0]
    log['G']={'n':len(y),'positive':int(sum(y)),'fitted':'G' in models}
    return models,log
def question(r,m):
    return int(np.argmax(m['Q'].logits(np.array([q['xq'] for q in r['questions']])))) if 'Q' in m else 0
def ask(q,m):
    if q['legal']==['a1']:return q['v1']
    if 'a1' not in q['legal']:return q['v0']
    use=True if 'A' not in m else float(m['A'].logits(np.array([q['xa']]))[0])>0
    return q['v1'] if use else q['v0']
def decide(r,m,bits=(1,1,1)):
    m={k:v for k,v in m.items() if bits[{'G':0,'Q':1,'A':2}[k]]}
    j=question(r,m);q=r['questions'][j]
    take=True if 'G' not in m else float(m['G'].logits(np.array([q['xq']]))[0])>0
    return (ask(q,m) if take else r['wait'],float(take))
def summary(rs,preds):
    values=np.array([[p[0] for p in x] for x in preds]);asks=np.array([[p[1] for p in x] for x in preds]);base=np.array([r['wait'] for r in rs]);delta=values.mean(0)-base
    ix=np.random.default_rng(42).integers(0,len(rs),(2000,len(rs)));ci=np.quantile(delta[ix].mean(1),[.025,.975])*100
    return {'n':len(rs),'quality':round(100*float(values.mean()),3),'seed_std':round(100*float(values.mean(1).std()),3),'wait_quality':round(100*float(base.mean()),3),
            'gain_pp':round(100*float(delta.mean()),3),'ci95_pp':[round(float(c),3) for c in ci],
            'ask_pct':round(100*float(asks.mean()),3),'improved_pct':round(100*float((values>base+1e-12).mean()),3),'degraded_pct':round(100*float((values<base-1e-12).mean()),3)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();start=time.time();groups={};raw=0;invalid=0
    for f in sorted(glob.glob(str(Path(a.teacher)/'clarifybench/choice/*/tasks/*.json'))):
        d=json.load(open(f));raw+=1;t=d.get('teacher') or {};b=d.get('forced_branch') or t.get('branch') or {}
        if d.get('status')!='measured' or t.get('branch_invalid') or b.get('cp')!=0:invalid+=1;continue
        tid=d['task_id'];v=groups.setdefault(tid,{'wait':None,'qs':{},'xq':None})
        if t.get('q_features'):v['xq']=t['q_features']
        if b.get('action')=='wait':v['wait']=d['Y']
        else:v['qs'].setdefault(b.get('q_index'),{})[b.get('a_choice')]={'Y':d['Y'],'xa':t.get('a_features'),'legal':t.get('legal') or []}
    records=[]
    for tid,d in groups.items():
        if d['wait'] is None or not d['xq']:continue
        qs=[]
        for j in range(len(d['xq'])):
            v=d['qs'].get(j,{})
            if 'a0' not in v or 'a1' not in v or v['a1']['xa'] is None:continue
            qs.append({'xq':d['xq'][j],'xa':v['a1']['xa'],'v0':v['a0']['Y'],'v1':v['a1']['Y'],'legal':v['a1']['legal']})
        if not qs:continue
        group=['D_A','D_Q','D_G'][int(h('hc_group:'+tid),16)%3]
        records.append({'task_id':tid,'group':group,'wait':d['wait'],'questions':qs})
    records.sort(key=lambda r:h('outer|'+r['task_id']));fold={r['task_id']:i%5 for i,r in enumerate(records)}
    bits=list(itertools.product([0,1],repeat=3));variants=['full','stale_labels','no_LA','no_LQ','no_LG'];pred={v:[[] for _ in range(3)] for v in variants};fac={''.join(map(str,b)):[[] for _ in range(3)] for b in bits};ordered=[];logs=[]
    for k in range(5):
        tr=[r for r in records if fold[r['task_id']]!=k];te=[r for r in records if fold[r['task_id']]==k];ordered+=te
        assert not ({r['task_id'] for r in tr}&{r['task_id'] for r in te})
        for si,seed in enumerate([42,123,456]):
            for v in variants:
                m,l=fit(tr,seed,v);pred[v][si]+=[decide(r,m) for r in te];logs.append({'fold':k,'seed':seed,'variant':v,'modules':l})
                if v=='full':
                    for b in bits:fac[''.join(map(str,b))][si]+=[decide(r,m,b) for r in te]
    result={'protocol':'Exploratory approximate-branch replay; environment states are not certified identical; excluded from reported controlled comparisons',
            'eligible_for_matched_state_claim':False,
            'raw_branch_records':raw,'filtered_records':invalid,'n_complete_checkpoints':len(records),'feature_dim':64,'new_api_calls':0,
            'factorial':{k:summary(ordered,v) for k,v in fac.items()},'training':{k:summary(ordered,v) for k,v in pred.items()},'fit_logs':logs,'elapsed_s':round(time.time()-start,2)}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'n':len(records),'full':result['training']['full'],'all_modules_fitted':all(all(m['fitted'] for m in l['modules'].values()) for l in logs if l['variant']=='full')}),flush=True)
if __name__=='__main__':main()
