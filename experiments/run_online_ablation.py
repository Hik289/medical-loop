"""Matched online module-ablation pilot using an existing benchmark runtime.

The original model files, task data, and run caches are never overwritten.
Runtime configuration and conversation records remain in the private run folder.
"""
import argparse,copy,hashlib,itertools,json,os,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
import numpy as np
from collections import Counter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--per-subset',type=int,default=2);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--request-timeout',type=int,default=90);ap.add_argument('--task-timeout',type=int,default=600)
    ap.add_argument('--retry-errors',action='store_true');a=ap.parse_args()
    root=Path(a.project_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'experiments/e1_methods'))
    from e1.config import load_config
    from e1.runner import task_universe,run_jobs
    from e1 import runner
    from e1.store import task_path,read_json
    # The inherited writer uses a process-wide temporary filename. Serialize
    # configuration writes only; independent API episodes still run concurrently.
    config_lock=Lock();resolve=runner.resolved_cfg_path
    def safe_resolve(config):
        with config_lock:return resolve(config)
    runner.resolved_cfg_path=safe_resolve
    cfg=load_config();cfg['llm'].update(request_timeout_s=a.request_timeout,max_retries=0,app_retries=0)
    cfg['run'].update(out_dir=str(out),concurrency=1,task_timeout_s=a.task_timeout,python=sys.executable,retry_errors=a.retry_errors)
    uni=task_universe(cfg,'clarifybench');chosen=[]
    for subset in ['A','E','I']:
        ids=[t for t,v in uni.items() if t.startswith(subset+'/') and v['split']=='test' and v['available']]
        ids.sort(key=lambda t:hashlib.sha256(('online-ablation|'+t).encode()).hexdigest());chosen+=ids[:a.per_subset]
    variants={'full':None,'no_G':'G','no_Q':'Q','no_A':'A','autonomous':None}
    original=root/'runs/e1_methods/hc_modules/choice/iter1'
    for v,disabled in variants.items():
        d=out/'models'/v;d.mkdir(parents=True,exist_ok=True)
        for name in ['G','Q','A']:
            x=json.load(open(original/f'{name}_s42.json'))
            if name==disabled:
                x['params']={k:np.zeros_like(np.asarray(val)).tolist() for k,val in x['params'].items()}
                x['params']['b3']=[0.0 if name=='Q' else 30.0]
            (d/f'{name}_s42.json').write_text(json.dumps(x))
    def task(tid):
        records={}
        # Serialize variants within a task so a shared prompt cache has one writer.
        for v in variants:
            c=copy.deepcopy(cfg);c['run']['out_dir']=str(out/v);c['methods']['humanchoice']['modules_dir']=str(out/'models'/v)
            method='autonomous' if v=='autonomous' else 'humanchoice';proto='native' if v=='autonomous' else 'choice'
            path=task_path(c['run']['out_dir'],'clarifybench',proto,method,tid)
            job={'benchmark':'clarifybench','method':method,'protocol':proto,'task_id':tid,'split':'test','out':path,
                 'replay_cache':str(out/'shared_prefix'/tid.replace('/','__'))}
            run_jobs(c,[job]);records[v]=read_json(path)
            print(json.dumps({'case_done':tid,'variant':v,'status':records[v].get('status')}),flush=True)
        return records
    rows=[];start=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(task,t) for t in chosen]):
            rows.append(f.result());(out/'progress.json').write_text(json.dumps({'complete_task_groups':len(rows),'total_task_groups':len(chosen),'elapsed_s':round(time.time()-start,1)}))
    common=[r for r in rows if all(r[v] and r[v].get('status')=='measured'
            and isinstance(r[v].get('metrics',{}).get('param_match_rate'),(int,float)) for v in variants)]
    result={'protocol':'Matched online pilot; hash-selected held-out tasks balanced by official request category; shared prefix cache; serialized variants per task; seed 42',
            'request_timeout_s':a.request_timeout,'task_timeout_s':a.task_timeout,'retry_errors':a.retry_errors,
            'selected_category_counts':dict(Counter(t.split('/')[0] for t in chosen)),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'api_count_scope':'Calls recorded in final execution files; incomplete timeout/retry ledgers are not a complete billing total',
            'n_selected':len(chosen),'n_complete_common':len(common),'variants':{},'elapsed_s':round(time.time()-start,2)}
    for v in variants:
        rs=[r[v] for r in rows];ok=[r for r in rs if r and r.get('status')=='measured']
        y=[r[v]['metrics']['param_match_rate'] for r in common];base=[r['autonomous']['metrics']['param_match_rate'] for r in common]
        delta=np.array(y)-np.array(base)
        ci=[None,None]
        if len(delta):
            ix=np.random.default_rng(42).integers(0,len(delta),(2000,len(delta)));ci=(100*np.quantile(delta[ix].mean(1),[.025,.975])).tolist()
        result['variants'][v]={'measured':len(ok),'errors':sum(bool(r) and r.get('status')=='error' for r in rs),'timeouts':sum(bool(r) and r.get('status')=='timeout' for r in rs),
            'common_n':len(common),'quality':100*float(np.mean(y)) if y else None,'gain_pp':100*float(delta.mean()) if len(delta) else None,'ci95_pp':ci,
            'questions_per_task':float(np.mean([r[v].get('controller',{}).get('counters',{}).get('questions',0) for r in common])) if common else None,
            'mean_wall_s':float(np.mean([r[v]['wall_s'] for r in common])) if common else None,
            'live_api_calls':sum(((r.get('api') or {}).get('total') or {}).get('live_calls',0) for r in rs if r)}
    (out/'online_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'online_complete':True,'summary':result}),flush=True)
if __name__=='__main__':main()
