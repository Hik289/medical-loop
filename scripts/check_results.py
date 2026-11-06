"""Check the arithmetic of the reported aggregates without record-level data."""
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def check(actual, expected, *, tolerance=0.015):
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise ValueError(f"Aggregate check failed: {actual} vs {expected}")


def main():
    e2 = load("E2_medical")
    base = e2["base_autonomous"]["record_acc"]
    hc = e2["E2"]["HumanChoice"]
    check(hc["rescue"] - hc["harm"], hc["net_rescue"])
    check(hc["record_acc"] - base, hc["net_rescue"])
    check(e2["n_tasks"], 270)
    if e2["simulated"] is not True:
        raise ValueError("E2's same-model simulated response label is missing")

    e5 = load("E5_expert_iteration_simulated")
    check(len(e5["iter_curve"]), 4)
    for i, (row, net_rescue) in enumerate(zip(e5["iter_curve"], [-1.11, 0, 0, 0])):
        check(row["iter"], i)
        check(row["test"]["net_rescue"]["mean"], net_rescue)
        check(row["test"]["final_quality"]["mean"] - base, net_rescue)

    ntb = load("E7_noisytoolbench")["agg"]
    check(ntb["autonomous"]["n"], 200)
    check(ntb["humanchoice"]["n"], 200)
    check(ntb["humanchoice"]["param_match_pct"] - ntb["autonomous"]["param_match_pct"], 32.83)

    e8 = load("E8_medical")["E8"]
    check(e8["HumanChoice 0% Availability"]["final_quality"], base)
    check(e8["HumanChoice 0% Availability"]["task_completion"], 100)
    print("Aggregate checks passed: E2, E5, E7, E8")


if __name__ == "__main__":
    main()
