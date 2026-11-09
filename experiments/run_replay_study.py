"""Task-grouped held-out replay, module interactions, and training-loss ablations.

Inputs are API-generated checkpoint records. This program makes no API calls and
never exports task text, identifiers, answers, or clinical fields in its summary.
"""
import argparse,copy,hashlib,itertools,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from replay_modules import policy as P,features as F
from replay_modules.mlp import train_bce,train_pairwise

SEEDS=(42,123,456)
def digest(x):
    return hashlib.sha256(str(x).encode()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def source(r):
    return str(r['task_id']).split('_',1)[-1]
def utility(r,o,lam):
    return float(o['Y'])-lam*float(o.get('seconds',0))
def evaluate(mods,recs,bits=(1,1,1)):
    g,q,a=bits
    m=P.Modules(A=mods.A if a else None,Q=mods.Q if q else None,G=mods.G if g else None)
    return [P.decide(m,r) for r in recs]
def report(recs,predictions,lam):
    assert all(len(x)==len(recs) for x in predictions)
    assert all(o.get('status')=='measured' for pred in predictions for o in pred)
    ys=np.array([[o['Y'] for o in pred] for pred in predictions],float)
    base=np.array([r['base']['Y'] for r in recs],float)
    sec=np.array([[o.get('seconds',0) for o in pred] for pred in predictions],float)
    ask=np.array([[bool(o['asked']) for o in pred] for pred in predictions])
    delta=ys.mean(0)-base
    groups=sorted({r['cluster_id'] for r in recs})
    gs=np.array([sum(delta[i] for i,r in enumerate(recs) if r['cluster_id']==g) for g in groups])
    gn=np.array([sum(r['cluster_id']==g for r in recs) for g in groups])
    rng=np.random.default_rng(42);ix=rng.integers(0,len(groups),(2000,len(groups)))
    ci=np.quantile(gs[ix].sum(1)/gn[ix].sum(1),[.025,.975])*100
    return {'n':len(recs),'n_clusters':len(groups),'quality':round(float(100*ys.mean()),3),
            'quality_seed_std':round(float(100*ys.mean(1).std()),3),
            'baseline_quality':round(float(100*base.mean()),3),
            'delta_pp':round(float(100*delta.mean()),3),'delta_ci95_pp':[round(float(v),3) for v in ci],
            'improved_pct':round(float(100*(ys>base+1e-12).mean()),3),
            'degraded_pct':round(float(100*(ys<base-1e-12).mean()),3),
            'asked_pct':round(float(100*ask.mean()),3),'estimated_seconds':round(float(sec.mean()),3),
            'utility':round(float((ys-lam*sec).mean()),5)}
def train(recs,seed,epochs,variant='full'):
    mods=P.Modules();log={}
    X,y,info=P.build_A(recs)
    if variant!='no_LA' and len(y)>=4:mods.A,tl=train_bce(X,y,seed,epochs);info.update(tl)
    log['A']={**info,'trained':mods.A is not None}
    teacher=P.Modules() if variant=='stale_labels' else mods
    xp,xn,info=P.build_Q(recs,teacher)
    if variant!='no_LQ' and len(xp)>=4:mods.Q,tl=train_pairwise(xp,xn,seed,epochs);info.update(tl)
    log['Q']={**info,'trained':mods.Q is not None}
    teacher=P.Modules() if variant=='stale_labels' else mods
    X,y,info=P.build_G(recs,teacher)
    if variant!='no_LG' and len(y)>=4:mods.G,tl=train_bce(X,y,seed,epochs);info.update(tl)
    log['G']={**info,'trained':mods.G is not None,'positive_targets':int(y.sum()) if len(y) else 0}
    return mods,log
def feature_audit(recs):
    n=0
    for r in recs:
        if not r.get('questions'):continue
        scores=[-float(i) for i in range(len(r['questions']))]
        for j,q in enumerate(r['questions']):
            alt=copy.deepcopy(r);alt['gold']='REDACTED';alt['base']['Y']=1-float(r['base']['Y'])
            altq=alt['questions'][j];altq['Y_ask']=123.;altq['answer']={'r_t':'changed','r_t_type':'skip'};altq['revision']={'action':'changed','changed':True}
            assert np.array_equal(F.q_feat(r,q),F.q_feat(alt,altq))
            assert np.array_equal(F.g_feat(r,q,scores,j),F.g_feat(alt,altq,scores,j))
            n+=1
    return {'questions_checked':n,'pre_answer_invariance_pass':True,'features':F.SCHEMA}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--medical-test',required=True);ap.add_argument('--medical-train',required=True)
    ap.add_argument('--medical-models',required=True);ap.add_argument('--ntb-records',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=30);args=ap.parse_args();t0=time.time()
    med=[r for r in read(args.medical_test) if r.get('status')=='measured']
    mt=[r for r in read(args.medical_train) if r.get('status')=='measured']
    assert not ({r['cluster_id'] for r in med}&{r['cluster_id'] for r in mt})
    raw=read(args.ntb_records);ntb=[r for r in raw if r.get('status')=='measured']
    for r in ntb:
        r['cluster_id']=source(r)
        r['group']=['D_A','D_Q','D_G'][int(digest('module|'+source(r)),16)%3]
    result={'protocol':'API-generated cached branches; fixed 30 epochs; no test-based selection; three seeds',
            'new_api_calls_in_this_program':0,'epochs':args.epochs,'seeds':list(SEEDS),
            'medical_train_test_disjoint':True,'ntb_coverage':dict(Counter(r.get('status') for r in raw)),
            'feature_audit':feature_audit(med+ntb),'medical_factorial':{},'medical_training':{},'ntb_factorial':{},'ntb_training':{},'ntb_cost':{},'ntb_pool':{},'fit_logs':[]}
    bits=list(itertools.product([0,1],repeat=3))
    frozen=[P.Modules.from_dict(read(Path(args.medical_models)/f'ei_iter3_seed{s}.json')) for s in SEEDS]
    result['medical_legacy_factorial']={}
    for b in bits:
        result['medical_legacy_factorial'][''.join(map(str,b))]=report(med,[evaluate(m,med,b) for m in frozen],P.LAMBDA_H)
    fresh_medical=[]
    for v in ['full','stale_labels','no_LA','no_LQ','no_LG']:
        pred=[]
        for seed in SEEDS:
            m,l=train(mt,seed,args.epochs,v);pred.append(evaluate(m,med));result['fit_logs'].append({'domain':'medical','variant':v,'seed':seed,'modules':l})
            if v=='full':fresh_medical.append(m)
        result['medical_training'][v]=report(med,pred,P.LAMBDA_H)
    for b in bits:
        result['medical_factorial'][''.join(map(str,b))]=report(med,[evaluate(m,med,b) for m in fresh_medical],P.LAMBDA_H)
    ids=sorted({source(r) for r in ntb},key=lambda k:digest('outer|'+k));fold={k:i%5 for i,k in enumerate(ids)}
    result['ntb_split']={'kind':'five-fold source-query-grouped cross-validation','n_source_queries':len(ids),'fold_source_counts':dict(Counter(fold.values())),
                         'no_source_overlap_between_train_test':True,'module_groups_source_disjoint':True}
    cv={v:[[] for _ in SEEDS] for v in ['full','stale_labels','no_LA','no_LQ','no_LG']}
    fact={''.join(map(str,b)):[[] for _ in SEEDS] for b in bits}
    pool={str(k):[[] for _ in SEEDS] for k in [1,2,3]}
    cost={str(lam):[[] for _ in SEEDS] for lam in [0.,.0005,.001]}
    ordered=[]
    for f in range(5):
        tr=[r for r in ntb if fold[source(r)]!=f];te=[r for r in ntb if fold[source(r)]==f];ordered+=te
        assert not ({source(r) for r in tr}&{source(r) for r in te})
        for si,seed in enumerate(SEEDS):
            full=None
            for v in cv:
                m,l=train(tr,seed,args.epochs,v);cv[v][si]+=evaluate(m,te)
                result['fit_logs'].append({'domain':'ntb','fold':f,'variant':v,'seed':seed,'modules':l})
                if v=='full':full=m
            for b in bits:fact[''.join(map(str,b))][si]+=evaluate(full,te,b)
            for k in [1,2,3]:
                subset=[dict(r,questions=r['questions'][:k]) for r in te];pool[str(k)][si]+=evaluate(full,subset)
            for lam in [0.,.0005,.001]:
                P.LAMBDA_H=lam
                m=full if lam==.0005 else train(tr,seed,args.epochs)[0]
                cost[str(lam)][si]+=evaluate(m,te)
            P.LAMBDA_H=.0005
        print(json.dumps({'finished_fold':f,'test_records':len(te),'elapsed_s':round(time.time()-t0,1)}),flush=True)
        write(Path(args.out)/'progress.json',{'finished_fold':f,'elapsed_s':round(time.time()-t0,1)})
    for k,pred in fact.items():result['ntb_factorial'][k]=report(ordered,pred,.0005)
    for k,pred in cv.items():result['ntb_training'][k]=report(ordered,pred,.0005)
    for k,pred in pool.items():result['ntb_pool'][k]=report(ordered,pred,.0005)
    for k,pred in cost.items():result['ntb_cost'][k]=report(ordered,pred,float(k))
    result['elapsed_seconds']=round(time.time()-t0,2)
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(Path(args.out)/'replay_results.json',result)
    print(json.dumps({'completed':True,'elapsed_seconds':result['elapsed_seconds'],'ntb_full':result['ntb_training']['full'],'medical_full':result['medical_training']['full']}),flush=True)
if __name__=='__main__':main()
