"""Compare request rankings at equal task-level request counts.

This overrides the default gate threshold for a budget diagnostic. Question
and action models are held fixed. No API calls or test-outcome selection.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from run_replay_study import read,write,train,digest,source,SEEDS
from replay_modules import policy as P,features as F

BUDGETS=(.1,.2,.4)
RULES=('random_expectation','uncertainty','learned_gate')

def predictions(model,records,fraction,rule):
    forced=[P.decide(model,r,variant='no_G') for r in records]
    eligible=[i for i,o in enumerate(forced) if o.get('asked')]
    k=min(round(len(records)*fraction),len(eligible));w=np.zeros(len(records))
    if rule=='random_expectation':
        if eligible:w[eligible]=k/len(eligible)
    else:
        scores=[]
        for i in eligible:
            r=records[i];qs=model.q_scores(r);qi=int(np.argmax(qs))
            value=model.p_ask(r,qi,qs) if rule=='learned_gate' else F._unc(r)
            scores.append((i,value,digest(r['task_id'])))
        scores.sort(key=lambda x:(-x[1],x[2]));w[[i for i,_,_ in scores[:k]]]=1
    base=np.array([r['base']['Y'] for r in records]);ask=np.array([o['Y'] for o in forced])
    seconds=np.array([o.get('seconds',0) for o in forced])
    return np.column_stack([base+w*(ask-base),w*(ask<base-1e-12),w,w*seconds])

def summarize(records,values):
    a=np.asarray(values);base=np.array([r['base']['Y'] for r in records]);delta=a[:,:,0].mean(0)-base
    groups=sorted({r['cluster_id'] for r in records})
    total=np.array([sum(delta[i] for i,r in enumerate(records) if r['cluster_id']==g) for g in groups])
    size=np.array([sum(r['cluster_id']==g for r in records) for g in groups])
    ix=np.random.default_rng(42).integers(0,len(groups),(2000,len(groups)))
    ci=100*np.quantile(total[ix].sum(1)/size[ix].sum(1),[.025,.975])
    return {'n':len(records),'quality':round(100*float(a[:,:,0].mean()),3),
            'harm_pct':round(100*float(a[:,:,1].mean()),3),'ask_pct':round(100*float(a[:,:,2].mean()),3),
            'estimated_seconds':round(float(a[:,:,3].mean()),3),
            'utility':round(float((a[:,:,0]-P.LAMBDA_H*a[:,:,3]).mean()),5),
            'delta_pp':round(100*float(delta.mean()),3),'delta_ci95_pp':[round(float(c),3) for c in ci]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--medical-train',required=True);ap.add_argument('--medical-test',required=True)
    ap.add_argument('--ntb-records',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();t=time.time()
    mt=read(a.medical_train);med=read(a.medical_test);ntb=[r for r in read(a.ntb_records) if r.get('status')=='measured']
    assert all(r.get('status')=='measured' for r in mt+med)
    assert not ({r['cluster_id'] for r in mt}&{r['cluster_id'] for r in med})
    for r in ntb:r.update(cluster_id=source(r),group=['D_A','D_Q','D_G'][int(digest('module|'+source(r)),16)%3])
    ids=sorted({source(r) for r in ntb},key=lambda k:digest('outer|'+k));fold={k:i%5 for i,k in enumerate(ids)}
    result={'protocol':'Equal request counts; fixed question/action models; thresholds overridden; random is an exact expected mixture; three seeds; source-grouped tool folds',
            'new_api_calls':0,'medical':{},'tools':{}}
    for domain in ['medical','tools']:
        pred={(b,r):[[] for _ in SEEDS] for b in BUDGETS for r in RULES};ordered=[]
        splits=[(mt,med)] if domain=='medical' else [([r for r in ntb if fold[source(r)]!=f],[r for r in ntb if fold[source(r)]==f]) for f in range(5)]
        for tr,te in splits:
            ordered+=te
            for si,seed in enumerate(SEEDS):
                model,_=train(tr,seed,30)
                for b,rule in pred:pred[b,rule][si].extend(predictions(model,te,b,rule).tolist())
        result[domain]={str(b):{rule:summarize(ordered,pred[b,rule]) for rule in RULES} for b in BUDGETS}
    result.update(elapsed_s=round(time.time()-t,2),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(a.out,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
