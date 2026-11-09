"""Verify epoch selections against complete trajectories and fixed-epoch results."""
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments/results/supplementary'

def main():
    d=json.loads((RESULTS/'epoch_selection.json').read_text())
    old=json.loads((RESULTS/'replay_results.json').read_text())
    rows=d['per_epoch'];assert [r['epoch'] for r in rows]==list(range(1,31))
    assert d['seeds']==[42,123,456] and d['new_api_calls']==0
    assert hashlib.sha256((ROOT/'experiments/run_epoch_sweep.py').read_bytes()).hexdigest()==d['code_sha256']
    selected=0
    for domain,families in d['envelope'].items():
        for family,items in families.items():
            for key,metrics in items.items():
                for name,item in metrics.items():
                    candidates={r['epoch']:r[domain][family][key][name] for r in rows}
                    target=(min if name in {'degraded_pct','asked_pct','estimated_seconds'} else max)(candidates.values())
                    assert math.isclose(target,item['value'],abs_tol=1e-8)
                    assert math.isclose(candidates[item['epoch']],item['value'],abs_tol=1e-8)
                    selected+=1
    comparisons=0
    for domain,prefix in [('medical','medical'),('tools','ntb')]:
        for family in ['factorial','training']+(['cost','pool'] if domain=='tools' else []):
            for key,new in rows[-1][domain][family].items():
                for metric in ['quality','degraded_pct','asked_pct','estimated_seconds','utility']:
                    tolerance=.00051 if metric!='utility' else .000006
                    assert math.isclose(new[metric],old[f'{prefix}_{family}'][key][metric],abs_tol=tolerance)
                    comparisons+=1
    print(f'Verified {len(rows)} epoch budgets, {selected} metric selections, and {comparisons} fixed-epoch values.')

if __name__=='__main__':main()
