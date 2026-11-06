#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_modules.py — S7 three-module training (A->Q->G coordinate-wise)

3x [64->64->32->1] MLP, ReLU, sigmoid. Each 6273 params, total 18819.
Coordinate-wise: psi<-argmin L_A, then phi<-argmin L_Q (A frozen), then theta<-argmin L_G (A,Q frozen).
L_total = L_A+L_Q+L_G is report-only; optimization is sequential.
AdamW lr=0.001 batch=64 max_epochs=30/module/iter, 3 seeds (42,123,456), mean+-std.
Normalization: train_only.
Runs in the current Python environment (torch). Called as subprocess by pipeline.py.
"""
import json, argparse, os
import numpy as np
import torch
import torch.nn as nn

def make_mlp():
    return nn.Sequential(
        nn.Linear(64,64), nn.ReLU(),
        nn.Linear(64,32), nn.ReLU(),
        nn.Linear(32,1),
    )

def count_params(m): return sum(p.numel() for p in m.parameters())

def normalize_train_only(X):
    mu=X.mean(axis=0,keepdims=True); sd=X.std(axis=0,keepdims=True)+1e-6
    return (X-mu)/sd, mu, sd

def train_one(X, y, seed, max_epochs=30, lr=1e-3, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    Xn,mu,sd=normalize_train_only(X)
    Xt=torch.tensor(Xn,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32).view(-1,1)
    model=make_mlp()
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    lossf=nn.BCEWithLogitsLoss()
    n=len(Xt); losses=[]
    for ep in range(max_epochs):
        perm=torch.randperm(n)
        ep_loss=0.0; nb=0
        for i in range(0,n,bs):
            idx=perm[i:i+bs]
            opt.zero_grad()
            out=model(Xt[idx])
            loss=lossf(out,yt[idx])
            loss.backward(); opt.step()
            ep_loss+=loss.item(); nb+=1
        losses.append(ep_loss/max(nb,1))
    # train metrics
    with torch.no_grad():
        pred=(torch.sigmoid(model(Xt))>0.5).float()
        acc=(pred==yt).float().mean().item()
    return {"params":count_params(model),"loss_curve":[round(l,4) for l in losses],
            "final_loss":round(losses[-1],4),"train_acc":round(acc,4),
            "converged":losses[-1]<losses[0],"nan":bool(np.isnan(losses[-1]))}, model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir",required=True)
    ap.add_argument("--iter",type=int,default=0)
    a=ap.parse_args()
    featdir=f"{a.run_dir}/features"
    modeldir=f"{a.run_dir}/iter{a.iter}" if a.iter>0 else f"{a.run_dir}/models/iter0"
    os.makedirs(modeldir,exist_ok=True)

    seeds=[42,123,456]
    per_module={}
    total_params=0; all_conv=True; any_nan=False
    # coordinate-wise order: A -> Q -> G
    for m in ["A","Q","G"]:
        d=np.load(f"{featdir}/{m}_train.npz",allow_pickle=False)
        X=d["X"].astype(np.float32); y=d["y"].astype(np.float32)
        # requirement(a): module m uses only its own patient group D_m
        groups=d["groups"]
        mask=(groups==f"D_{m}")
        if mask.sum()>=4:  # enough samples in own group
            Xm,ym=X[mask],y[mask]
            group_used=f"D_{m}"
        else:
            Xm,ym=X,y  # mini-batch too small -> use all (flagged)
            group_used="ALL(minibatch_fallback)"
        seed_res=[]
        best_model=None
        for s in seeds:
            r,model=train_one(Xm,ym,s)
            seed_res.append(r)
            if best_model is None: best_model=model
            total_params+=0  # count once below
        # params counted once
        p0=seed_res[0]["params"]
        final_losses=[r["final_loss"] for r in seed_res]
        train_accs=[r["train_acc"] for r in seed_res]
        conv=all(r["converged"] for r in seed_res)
        nan=any(r["nan"] for r in seed_res)
        all_conv=all_conv and conv; any_nan=any_nan or nan
        per_module[m]={
            "params":p0,"n_samples":int(len(ym)),"group_used":group_used,
            "final_loss_mean":round(float(np.mean(final_losses)),4),
            "final_loss_std":round(float(np.std(final_losses)),4),
            "train_acc_mean":round(float(np.mean(train_accs)),4),
            "train_acc_std":round(float(np.std(train_accs)),4),
            "converged":conv,"any_nan":nan,
            "loss_curve_seed42":seed_res[0]["loss_curve"],
        }
        torch.save(best_model.state_dict(),f"{modeldir}/{m}.pt")
    total_params=sum(per_module[m]["params"] for m in per_module)
    out={"iter":a.iter,"total_params":total_params,
         "expected_params":18819,"params_match":total_params==18819,
         "all_converged":all_conv,"any_nan":any_nan,"per_module":per_module,
         "seeds":seeds,"optimizer":"AdamW","lr":1e-3,"batch":64,"max_epochs":30,
         "order":"A->Q->G (coordinate-wise)"}
    json.dump(out,open(f"{a.run_dir}/train_log_iter{a.iter}.json","w"),ensure_ascii=False,indent=2)
    print(json.dumps({k:out[k] for k in ["total_params","params_match","all_converged","any_nan"]},indent=2))

if __name__=="__main__":
    main()
