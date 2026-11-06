#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anchor2_assert.py — H0.anchor_2 (debug_move=ASSERT) 评估代码正确性

构造已知 toy ground truth, 断言 eval_metrics 输出 == 手算值。
覆盖: Record Accuracy / macro-F1 / per-class recall / Rescue / Harm / Net Rescue
     + 要求(b): 证据不足→正确标 Unresolved(高分) vs 证据不足→过度断言(Unsupported)
全部 assertion 通过 → anchor_2 PASS; 任一 fail → 硬 fail 检查断言明细。

纯本地, 无 API 调用。
"""
import sys, json, math
from datetime import datetime, timezone, timedelta
import eval_metrics as EM
from runtime import runs_dir

JST = timezone(timedelta(hours=9))
PASS, FAIL = [], []

def check(name, got, want, tol=1e-9):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and want == want else (got == want)
    (PASS if ok else FAIL).append((name, got, want))
    tag = "✅" if ok else "❌"
    print(f"  {tag} {name}: got={got} want={want}")

print("=" * 70)
print("H0.anchor_2 评估代码 ASSERT — 构造 toy GT, 对比手算值")
print("=" * 70)

# ---------------------------------------------------------------------------
# Test 1: Record Accuracy / macro-F1 / per-class recall
# ---------------------------------------------------------------------------
# 构造 10 样本, 真值 = A A A B B B C C C C (A=3, B=3, C=4)
# 预测          = A A B B B A C C A C
# 手算:
#   正确: idx0(A✓) idx1(A✓) idx2(A→B✗) idx3(B✓) idx4(B✓) idx5(B→A✗)
#         idx6(C✓) idx7(C✓) idx8(C→A✗) idx9(C✓)  → 正确 7/10 = 0.7
#   recall A: 真A=3{0,1,2}, 预测对=2{0,1} → 2/3
#   recall B: 真B=3{3,4,5}, 预测对=2{3,4} → 2/3
#   recall C: 真C=4{6,7,8,9}, 预测对=3{6,7,9} → 3/4
print("\n[Test 1] record_accuracy")
gold = ["A","A","A","B","B","B","C","C","C","C"]
pred = ["A","A","B","B","B","A","C","C","A","C"]
r = EM.record_accuracy(pred, gold)
check("overall_acc", r["overall_acc"], 0.7)
check("recall_A", r["per_class_recall"]["A"], 2/3, tol=1e-4)   # per_class_recall 舍入到 4 位
check("recall_B", r["per_class_recall"]["B"], 2/3, tol=1e-4)
check("recall_C(AKI)", r["per_class_recall"]["C"], 3/4)
check("aki_recall", r["aki_recall"], 3/4)
check("support_C", r["support"]["C"], 4)
# macro-F1 手算 (pred = A A B B B A C C A C):
#  A: pred_pos={0,1,5,8}=4, tp(gold=A)={0,1}=2 → prec=2/4=0.5, rec=2/3 → f1=2*0.5*(2/3)/(0.5+2/3)=4/7
#  B: pred_pos={2,3,4}=3, tp={3,4}=2 → prec=2/3, rec=2/3 → f1=2/3
#  C: pred_pos={6,7,9}=3, tp={6,7,9}=3 → prec=1, rec=3/4 → f1=2*1*0.75/1.75=0.857142857
f1_A = 2*0.5*(2/3)/(0.5+2/3); f1_B = 2/3; f1_C = 2*1.0*0.75/(1.0+0.75)
check("macro_f1", r["macro_f1"], (f1_A+f1_B+f1_C)/3)

# ---------------------------------------------------------------------------
# Test 2: Rescue / Harm / Net Rescue (已知 Base 对/错 × HC 对/错 配对)
# ---------------------------------------------------------------------------
# 8 配对:
#  base: T T T F F F T F
#  hc:   T F F T T F F T
# 分类: (T,T)=no-change | (T,F)=harm | (T,F)=harm | (F,T)=rescue | (F,T)=rescue | (F,F)=no-change | (T,F)=harm | (F,T)=rescue
#  rescue(F→T) = idx3,4,7 = 3/8 = 0.375
#  harm(T→F)   = idx1,2,6 = 3/8 = 0.375
#  net = 0
print("\n[Test 2] rescue_harm")
base = [True, True, True, False, False, False, True, False]
hc   = [True, False, False, True, True, False, False, True]
rh = EM.rescue_harm(base, hc)
check("rescue", rh["rescue"], 3/8)
check("harm", rh["harm"], 3/8)
check("net_rescue", rh["net_rescue"], 0.0)
check("n_pairs", rh["n_pairs"], 8)
check("n_rescue", rh["n_rescue"], 3)
check("n_harm", rh["n_harm"], 3)

# Test 2b: 按类分层 (AKI=C)
# gold: A A C C C B C B  → C 的位置 idx2,3,4,6
#   base[C] = T F F T ; hc[C] = F T T F
#   idx2:(T,F)=harm idx3:(F,T)=rescue idx4:(F,T)=rescue idx6:(T,F)=harm
#   rescue=2/4=0.5 harm=2/4=0.5 net=0
print("\n[Test 2b] rescue_harm stratified AKI=C")
gold8 = ["A","A","C","C","C","B","C","B"]
rhc = EM.rescue_harm(base, hc, gold=gold8, stratify_class="C")
check("aki_rescue", rhc["rescue"], 0.5)
check("aki_harm", rhc["harm"], 0.5)
check("aki_net", rhc["net_rescue"], 0.0)
check("aki_n_pairs", rhc["n_pairs"], 4)

# ---------------------------------------------------------------------------
# Test 3: 要求(b) — Evidence Review + Unsupported Inference
# ---------------------------------------------------------------------------
# 4 条 toy review:
#  c0: 证据不足(allow_unresolved=T), pred=Unresolved      → 正确 score=1 unsup=0
#  c1: 证据不足(allow_unresolved=T), pred="CKD确诊"(断言)   → 过度断言 score=0 unsup=1
#  c2: 证据充足(allow_unresolved=F, acceptable={AKI}), pred=AKI → score=1 unsup=0
#  c3: 证据充足(allow_unresolved=F, acceptable={AKI}), pred=Unresolved → 弃判 score=0 unsup=0
# 手算: mean score = (1+0+1+0)/4 = 0.5 ; unsupported_rate = (0+1+0+0)/4 = 0.25
print("\n[Test 3] evidence_review + unsupported_inference (要求(b))")
preds = ["Unresolved", "CKD确诊", "AKI", "Unresolved"]
refs = [
    {"allow_unresolved": True, "acceptable": []},
    {"allow_unresolved": True, "acceptable": []},
    {"allow_unresolved": False, "acceptable": ["AKI"]},
    {"allow_unresolved": False, "acceptable": ["AKI"]},
]
# 逐条断言
s0 = EM.score_review(preds[0], refs[0]); check("c0_unresolved_score", s0["score"], 1.0); check("c0_unsup", s0["unsupported"], 0)
s1 = EM.score_review(preds[1], refs[1]); check("c1_overclaim_score", s1["score"], 0.0); check("c1_unsup(过度断言)", s1["unsupported"], 1)
s2 = EM.score_review(preds[2], refs[2]); check("c2_correct_score", s2["score"], 1.0); check("c2_unsup", s2["unsupported"], 0)
s3 = EM.score_review(preds[3], refs[3]); check("c3_abstain_score", s3["score"], 0.0); check("c3_unsup(弃判非过度)", s3["unsupported"], 0)
agg = EM.evidence_review_aggregate(preds, refs)
check("mean_review_score", agg["evidence_review_score"], 0.5)
check("unsupported_rate", agg["unsupported_inference_rate"], 0.25)

# ---------------------------------------------------------------------------
# Test 4: bootstrap CI sanity (确定性构造)
# ---------------------------------------------------------------------------
# 全 rescue (base 全错, hc 全对) → net 逐样本全=1 → CI = [1,1]
print("\n[Test 4] net_rescue_bootstrap_ci sanity")
b_all_fail = [False]*20; h_all_ok = [True]*20; clus = list(range(20))
ci = EM.net_rescue_bootstrap_ci(b_all_fail, h_all_ok, clus, n_boot=500)
check("all_rescue_point", ci["point"], 1.0)
check("all_rescue_ci_lo", ci["ci_lo"], 1.0)
# 全 no-change (base==hc) → net 逐样本全=0 → CI=[0,0], 下界不>0
b_eq = [True]*20; h_eq = [True]*20
ci0 = EM.net_rescue_bootstrap_ci(b_eq, h_eq, clus, n_boot=500)
check("no_change_point", ci0["point"], 0.0)
check("no_change_ci_lo", ci0["ci_lo"], 0.0)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"结果: {len(PASS)} PASS / {len(FAIL)} FAIL")
result = {
    "anchor": "H0.anchor_2",
    "timestamp": datetime.now(JST).isoformat(),
    "n_pass": len(PASS), "n_fail": len(FAIL),
    "status": "PASS" if not FAIL else "FAIL",
    "failures": [{"name": n, "got": g, "want": w} for n, g, w in FAIL],
}
runs_dir().mkdir(parents=True, exist_ok=True)
with open(runs_dir() / "anchor2_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"saved anchor2_result.json → status={result['status']}")
if FAIL:
    print("❌ anchor_2 FAIL — 硬 fail, 检查断言明细")
    sys.exit(1)
print("✅ anchor_2 PASS")
