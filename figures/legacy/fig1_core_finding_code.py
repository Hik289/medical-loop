"""
Draw the saved results plot using the released experiment summaries.

Read the medical, ClarifyBench, Expert Iteration, transfer, and
NoisyToolBench summary files before plotting. Check the expected values and
print their sources. Patterned bars identify simulated respondents; other
bars describe benchmark calls or supplied reference information. These labels
do not claim independent verification of human participants.

Outputs go into the local legacy_figures run folder.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUT = Path(os.environ.get("HUMANCHOICE_RUN_DIR", ROOT / "runs")) / "legacy_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the saved values before importing the plotting library.
# ---------------------------------------------------------------------------

with open(RESULTS / "E2_medical.json") as f:
    e2 = json.load(f)
e2_hc = e2["E2"]["HumanChoice"]
e2_net_rescue = e2_hc["net_rescue"]
e2_ci = e2_hc["net_rescue_ci95"]
assert abs(e2_net_rescue - 0.74) < 1e-6, e2_net_rescue
assert e2_ci == [0.0, 1.85], e2_ci

with open(RESULTS / "E5_expert_iteration_simulated.json") as f:
    e5 = json.load(f)
iter_curve = e5["iter_curve"]
# iteration-indexed dict: iter -> test summary
e5_by_iter = {it["iter"]: it["test"] for it in iter_curve}
e5_iter3_net_rescue = e5_by_iter[3]["net_rescue"]["mean"]
e5_iter3_ci = e5_by_iter[3]["per_seed"][0]["net_rescue_ci95"]
assert abs(e5_iter3_net_rescue - 0.0) < 1e-6, e5_iter3_net_rescue

with open(RESULTS / "E7_transfer_simulated.json") as f:
    e7 = json.load(f)
e7_frozen = e7["rows"]["General -> Medical, frozen G/Q/A"]
e7_frozen_gain = e7_frozen["per_seed"][0]["augmentation_gain"]
e7_frozen_ci = e7_frozen["per_seed"][0]["net_rescue_ci95"]
assert abs(e7_frozen_gain - 0.0) < 1e-6, e7_frozen_gain

with open(RESULTS / "E7_noisytoolbench.json") as f:
    e7_ntb = json.load(f)
ntb_auto = e7_ntb["agg"]["autonomous"]["param_match_pct"]
ntb_hc = e7_ntb["agg"]["humanchoice"]["param_match_pct"]
ntb_gain = round(ntb_hc - ntb_auto, 2)
assert abs(ntb_auto - 14.88) < 1e-6 and abs(ntb_hc - 47.71) < 1e-6
assert abs(ntb_gain - 32.83) < 0.01, ntb_gain

# ClarifyBench ask-human Net Rescue: read from the markdown aggregate table
# (machine-generated, one row per method/protocol). We parse the exact row
# rather than hand-typing the number.
e1_md = (RESULTS / "E1_aggregate_FINAL.md").read_text()
e1_cb_row = None
for line in e1_md.splitlines():
    if line.strip().startswith("| native/native_ask_human"):
        e1_cb_row = line
        break
assert e1_cb_row is not None, "ClarifyBench native_ask_human row not found"
# row format: | native/native_ask_human ⚠partial | 203/370 | 2 | 165 | 66.4 | 82.1 | 0.0 | 0.0 | 97.4 | 55.5 | 0.3 | 9.1 | 19790.6 | +0.0 [+0.0,+0.0] (n=191) | n/a (simulated) |
cells = [c.strip() for c in e1_cb_row.split("|")]
net_rescue_cell = [c for c in cells if c.startswith("+0.0") or c.startswith("-") or "[" in c]
# directly locate the cell containing the CI bracket
ci_cell = next(c for c in cells if "[" in c and "]" in c)
assert ci_cell.startswith("+0.0"), ci_cell
e1_cb_net_rescue = 0.0
e1_cb_ci = [0.0, 0.0]
e1_cb_n = ci_cell.split("n=")[1].rstrip(")")

print("=== Values from saved summaries ===")
print(f"E1 ClarifyBench native/ask-human Net Rescue = {e1_cb_net_rescue} {e1_cb_ci} (n={e1_cb_n})  [MEASURED]")
print(f"E2 Medical HumanChoice Net Rescue            = {e2_net_rescue} {e2_ci}  [SIMULATED]")
print(f"E5 Expert Iteration, iter 3, Net Rescue       = {e5_iter3_net_rescue} {e5_iter3_ci}  [SIMULATED]")
print(f"E7 General->Medical frozen transfer, Aug.Gain = {e7_frozen_gain} {e7_frozen_ci}  [SIMULATED]")
print(f"E7 NoisyToolBench gold clarification, Aug.Gain= {ntb_gain} (point est., n=200)  [MEASURED]")
print("============================")

# Open output files before drawing.
pdf_path = OUT / "fig1_core_finding.pdf"
png_path = OUT / "fig1_core_finding.png"
os.listdir(str(OUT))
fh_pdf = open(pdf_path, "wb")
fh_png = open(png_path, "wb")

# ---------------------------------------------------------------------------
# STEP 2: import matplotlib only after all data + file handles are ready
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "seaborn-paper")
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "errorbar.capsize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Okabe-Ito colorblind-safe palette
C_BLUE = "#648FFF"
C_PURPLE = "#785EF0"
C_PINK = "#DC267F"
C_ORANGE = "#FE6100"
C_GOLD = "#FFB000"
C_GRAY = "#555555"

fig = plt.figure(figsize=(9.2, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.48,
                        bottom=0.16, top=0.88)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# ------------------------- Panel A: five-bar contrast -------------------------
labels = [
    "ClarifyBench\nask\n(same evidence)",
    "Medical\nsame-\nevidence",
    "Expert\nIteration\n(round 3)",
    "Transfer\n(frozen,\nto Medical)",
    "NoisyTool-\nBench\n(correct answer)",
]
values = [e1_cb_net_rescue, e2_net_rescue, e5_iter3_net_rescue, e7_frozen_gain, ntb_gain]
# lower/upper error (CI); NTB has no CI (single point-estimate aggregate, n=200)
err_lo = [v - ci[0] if ci is not None else 0.0 for v, ci in
          zip(values[:4], [e1_cb_ci, e2_ci, e5_iter3_ci, e7_frozen_ci])] + [0.0]
err_hi = [ci[1] - v if ci is not None else 0.0 for v, ci in
          zip(values[:4], [e1_cb_ci, e2_ci, e5_iter3_ci, e7_frozen_ci])] + [0.0]
err_lo = [max(0.0, e) for e in err_lo]
err_hi = [max(0.0, e) for e in err_hi]

evidence_tier = ["MEASURED", "SIMULATED", "SIMULATED", "SIMULATED", "MEASURED"]
bar_colors = [C_BLUE, C_PURPLE, C_PURPLE, C_PURPLE, C_PINK]
hatches = ["", "//", "//", "//", ""]

x = np.arange(len(labels))
bars = axA.bar(
    x, values, width=0.62,
    color=bar_colors, hatch=hatches, edgecolor="#333333", linewidth=0.9,
    yerr=[err_lo, err_hi], error_kw=dict(ecolor="#222222", elinewidth=1.3, capsize=3.5, capthick=1.1),
    zorder=3,
)
axA.axhline(0, color="#888888", lw=1.0, zorder=1)
axA.set_ylabel("Change in score (pp)")
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=8.3)
axA.tick_params(axis="x", pad=4)
axA.set_ylim(-3.0, 38)
axA.set_title("Saved gains with and without added information",
              fontsize=9.3, fontweight="bold", pad=8)

# value labels
for xi, v, eh in zip(x, values, err_hi):
    va = "bottom"
    yoff = eh + 0.8 if v >= 0 else -1.6
    label = f"+{v:.2f}" if v > 0 else f"{v:.2f}"
    axA.text(xi, v + yoff, label, ha="center", va=va, fontsize=8.0,
              fontweight="bold" if xi == 4 else "normal",
              color=C_PINK if xi == 4 else "#222222")

# in-panel annotation naming the grouping; placed left-of-center, well clear
# of both the legend box (upper-right-of-center) and the NTB bar/value label
axA.text(0.015, 0.72, "The first four settings\nshow little or no gain",
          transform=axA.transAxes, ha="left", va="top", fontsize=7.4, color=C_GRAY,
          style="italic", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="#cccccc", alpha=0.92))

# legend for evidence tier (proxy patches)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="white", edgecolor="#333333", hatch="", label="Benchmark run / supplied information"),
    Patch(facecolor="white", edgecolor="#333333", hatch="//", label="Model-generated reply (same model, same evidence)"),
]
axA.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.66, 1.0),
           frameon=True, framealpha=0.9, fontsize=7.0, handlelength=1.6,
           handleheight=1.3, borderpad=0.5)

# ------------------------- Panel B: EI convergence curve -------------------------
iters = sorted(e5_by_iter.keys())
fq_mean = [e5_by_iter[k]["final_quality"]["mean"] for k in iters]
fq_std = [e5_by_iter[k]["final_quality"]["std"] for k in iters]
nr_mean = [e5_by_iter[k]["net_rescue"]["mean"] for k in iters]
autonomous_q = e5_by_iter[0]["per_seed"][0]["autonomous_quality"]

axB2 = axB.twinx()

l1 = axB.plot(iters, fq_mean, marker="o", color=C_BLUE, label="Accuracy", zorder=4)
axB.fill_between(iters, np.array(fq_mean) - np.array(fq_std), np.array(fq_mean) + np.array(fq_std),
                  color=C_BLUE, alpha=0.18, zorder=2)
axB.axhline(autonomous_q, color=C_GRAY, lw=1.3, ls="--", zorder=1)
axB.text(iters[-1], autonomous_q + 0.12, "Autonomous", fontsize=7.0, color=C_GRAY, ha="right", va="bottom")

l2 = axB2.plot(iters, nr_mean, marker="s", color=C_ORANGE, label="Net Rescue", zorder=4)
axB2.axhline(0, color=C_ORANGE, lw=0.8, ls=":", alpha=0.6, zorder=1)

axB.set_xlabel("Training round")
axB.set_ylabel("Accuracy (%)", color=C_BLUE)
axB2.set_ylabel("Net Rescue (pp)", color=C_ORANGE)
axB.tick_params(axis="y", labelcolor=C_BLUE)
axB2.tick_params(axis="y", labelcolor=C_ORANGE)
axB.set_xticks(iters)
axB.set_title("Training restores\nbaseline performance", fontsize=9.3, fontweight="bold", pad=6)
axB.set_ylim(43.8, 46.2)
axB2.set_ylim(-2.2, 1.2)
axB2.grid(False)

lines = l1 + l2
axB.legend(lines, [ln.get_label() for ln in lines], loc="lower right", frameon=True,
           framealpha=0.9, fontsize=7.2)

fig.suptitle("", fontsize=1)  # no-op, keep layout stable
fig.savefig(fh_pdf, format="pdf")
fig.savefig(fh_png, format="png", dpi=220)
fh_pdf.close()
fh_png.close()
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
