"""Retrospective epoch envelopes with complete, auditable score trajectories.

Every candidate refits A, then Q, then G for the stated number of epochs per
coordinate update. Each displayed metric may select a different epoch. These
test-set maxima/minima are descriptive envelopes, not locked test estimates.
Original fixed-epoch results remain separate. No API calls or raw-text exports.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,itertools,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from run_replay_study import read,write,train,evaluate,digest,source,SEEDS
from run_budget_study import predictions as budget_predictions,BUDGETS,RULES
from replay_modules import policy as P

VARIANTS=('full','stale_labels','no_LA','no_LQ','no_LG')
BITS=list(itertools.product([0,1],repeat=3))
DATA=None

def initialize(medical_train,medical_test,tool_records):
    global DATA
    mt=read(medical_train);med=read(medical_test)
    ntb=[r for r in read(tool_records) if r.get('status')=='measured']
    assert all(r.get('status')=='measured' for r in mt+med)
    assert not ({r['cluster_id'] for r in mt}&{r['cluster_id'] for r in med})
    for r in ntb:r.update(cluster_id=source(r),group=['D_A','D_Q','D_G'][int(digest('module|'+source(r)),16)%3])
    ids=sorted({source(r) for r in ntb},key=lambda k:digest('outer|'+k))
    fold={k:i%5 for i,k in enumerate(ids)}
    DATA=(mt,med,ntb,fold)

def summarize(records,values,lam=.0005):
    a=np.asarray(values);base=np.array([r['base']['Y'] for r in records])
    return {'n':len(records),'quality':round(100*float(a[:,:,0].mean()),6),
            'quality_seed_std':round(100*float(a[:,:,0].mean(1).std()),6),
            'baseline_quality':round(100*float(base.mean()),6),
            'delta_pp':round(100*float((a[:,:,0]-base).mean()),6),
            'improved_pct':round(100*float(a[:,:,4].mean()),6),
            'degraded_pct':round(100*float(a[:,:,1].mean()),6),
            'asked_pct':round(100*float(a[:,:,2].mean()),6),
            'estimated_seconds':round(float(a[:,:,3].mean()),6),
            'utility':round(float((a[:,:,0]-lam*a[:,:,3]).mean()),8)}

def array(records,preds):
    return [[o['Y'],float(o['Y']<r['base']['Y']-1e-12),float(o.get('asked',False)),o.get('seconds',0),
             float(o['Y']>r['base']['Y']+1e-12)] for r,o in zip(records,preds)]

def medical_scores(records,preds):
    gold=np.array([r['gold'] for r in records]);out=[]
    for predictions in preds:
        labels=np.array([o['action'] for o in predictions])
        assert all(abs(float(o['Y'])-float(o['action']==r['gold']))<1e-9 for r,o in zip(records,predictions))
        f1=[];recall=[]
        for lab in ['A','B','C']:
            tp=float(((labels==lab)&(gold==lab)).sum());pos=float((labels==lab).sum());support=float((gold==lab).sum())
            f1.append(2*tp/(pos+support) if pos+support else 0);recall.append(tp/support if support else 0)
        out.append([np.mean(f1),*recall])
    x=np.array(out).mean(0)*100
    return {'macro_f1':round(float(x[0]),6),'recall_A':round(float(x[1]),6),
            'recall_B':round(float(x[2]),6),'aki_recall':round(float(x[3]),6)}

def one_epoch(epoch):
    mt,med,ntb,fold=DATA;result={'epoch':epoch,'medical':{},'tools':{}}
    for domain in ['medical','tools']:
        families={}
        def target(family,key):
            return families.setdefault(family,{}).setdefault(key,[[] for _ in SEEDS])
        ordered=[];med_predictions={}
        splits=[(mt,med)] if domain=='medical' else [([r for r in ntb if fold[source(r)]!=f],[r for r in ntb if fold[source(r)]==f]) for f in range(5)]
        for tr,te in splits:
            ordered+=te
            for si,seed in enumerate(SEEDS):
                P.LAMBDA_H=.0005;full=None
                for variant in VARIANTS:
                    model,_=train(tr,seed,epoch,variant);pred=evaluate(model,te)
                    target('training',variant)[si].extend(array(te,pred))
                    if domain=='medical':med_predictions.setdefault(('training',variant),[[] for _ in SEEDS])[si].extend(pred)
                    if variant=='full':full=model
                for bits in BITS:
                    key=''.join(map(str,bits));pred=evaluate(full,te,bits)
                    target('factorial',key)[si].extend(array(te,pred))
                    if domain=='medical':med_predictions.setdefault(('factorial',key),[[] for _ in SEEDS])[si].extend(pred)
                for b in BUDGETS:
                    for rule in RULES:
                        a=budget_predictions(full,te,b,rule)
                        # Expected random selection has fractional harm/request entries.
                        base=np.array([r['base']['Y'] for r in te])
                        forced=np.array([P.decide(full,r,variant='no_G')['Y'] for r in te])
                        improved=a[:,2]*(forced>base+1e-12)
                        target('budget',f'{b}|{rule}')[si].extend(np.column_stack([a,improved]).tolist())
                if domain=='tools':
                    for k in [1,2,3]:
                        subset=[dict(r,questions=r['questions'][:k]) for r in te]
                        target('pool',str(k))[si].extend(array(te,evaluate(full,subset)))
                    for lam in [0.,.0005,.001]:
                        P.LAMBDA_H=lam;model=full if lam==.0005 else train(tr,seed,epoch)[0]
                        target('cost',str(lam))[si].extend(array(te,evaluate(model,te)))
                    P.LAMBDA_H=.0005
        for family,items in families.items():
            result[domain][family]={key:summarize(ordered,val,float(key) if family=='cost' else .0005) for key,val in items.items()}
            if domain=='medical' and family in ('training','factorial'):
                for key in items:result[domain][family][key].update(medical_scores(ordered,med_predictions[family,key]))
    return result

def envelopes(rows):
    lower={'degraded_pct','asked_pct','estimated_seconds'};out={}
    for domain in ['medical','tools']:
        out[domain]={}
        for family,items in rows[0][domain].items():
            out[domain][family]={}
            for key,metrics in items.items():
                selected={}
                for metric in metrics:
                    if metric in ['n','baseline_quality','quality_seed_std']:continue
                    candidates=[(r['epoch'],r[domain][family][key][metric]) for r in rows]
                    best=(min if metric in lower else max)(v for _,v in candidates)
                    ties=[e for e,v in candidates if abs(v-best)<1e-8]
                    selected[metric]={'value':best,'epoch':min(ties),'tied_epochs':ties}
                out[domain][family][key]=selected
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--medical-train',required=True);ap.add_argument('--medical-test',required=True)
    ap.add_argument('--ntb-records',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--max-epoch',type=int,default=30);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();start=time.time()
    rows=[];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.medical_train,a.medical_test,a.ntb_records)) as ex:
        futures={ex.submit(one_epoch,e):e for e in range(1,a.max_epoch+1)}
        for f in as_completed(futures):
            row=f.result();rows.append(row);write(out/f'epoch_{row["epoch"]:02}.json',row)
            print(json.dumps({'epoch_done':row['epoch'],'completed':len(rows),'elapsed_s':round(time.time()-start,1)}),flush=True)
    rows.sort(key=lambda x:x['epoch']);result={'selection':'Retrospective per-metric test-set epoch envelope; distinct cells may use different epochs; not locked test performance',
        'epoch_definition':'Epochs per A/Q/G coordinate update with refreshed downstream targets',
        'epochs':list(range(1,a.max_epoch+1)),'seeds':list(SEEDS),'new_api_calls':0,
        'medical_n':270,'tool_n':150,'tool_source_groups':50,'tool_protocol':'Five-fold source-query-grouped evaluation',
        'per_epoch':rows,'envelope':envelopes(rows),'coherent_best_quality':{},
        'elapsed_s':round(time.time()-start,2),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for domain in ['medical','tools']:
        e=max(rows,key=lambda r:(r[domain]['training']['full']['quality'],r[domain]['training']['full']['utility'],-r['epoch']))
        result['coherent_best_quality'][domain]={'epoch':e['epoch'],**e[domain]['training']['full']}
    write(out/'epoch_selection.json',result)
    print(json.dumps({'complete':True,'best_quality':result['coherent_best_quality'],'elapsed_s':result['elapsed_s']}),flush=True)
if __name__=='__main__':main()
