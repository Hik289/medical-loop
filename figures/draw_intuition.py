"""Render the archived programmatic schematic without replacing the README artwork."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as PlotPath

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})

INK, MUTED, RULE = "#182737", "#47586B", "#D9E2EB"
BLUE, SKY, BLUE_LINE = "#20578C", "#EEF6FD", "#96BCDE"
PURPLE, LILAC, PURPLE_LINE = "#66448E", "#F5EFFB", "#B99AD4"
AMBER, CREAM, AMBER_LINE = "#8A511E", "#FEF5E8", "#DDB178"
GREEN, MINT, GREEN_LINE = "#246951", "#EDF8F0", "#85B99D"
NEUTRAL = "#F5F8FB"

WIDTH, HEIGHT = 16.0, 5.80
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
ax.axis("off")


def label(x, y, s, size=13, color=INK, *, align="center", weight="normal", serif=False):
    return ax.text(x, y, s, fontsize=size, color=color, ha=align, va="center",
                   fontweight=weight, family="DejaVu Serif" if serif else "DejaVu Sans",
                   zorder=8)


def card(x, y, w, h, fill, edge, *, r=.13, lw=1.05, z=2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                           facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def line(points, color=BLUE, *, lw=1.45, dashed=False, z=5):
    ax.plot(*zip(*points), color=color, lw=lw, solid_capstyle="round",
            ls=(0, (3, 3)) if dashed else "-", zorder=z)


def arrow(points, color=BLUE, *, lw=1.45, dashed=False, scale=12):
    if len(points) > 2:
        line(points[:-1], color, lw=lw, dashed=dashed)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=scale, color=color, lw=lw,
                                 linestyle=(0, (3, 3)) if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=6))


def curve_arrow(start, bends, end, color=BLUE, *, lw=1.45):
    path = PlotPath([start, *bends, end],
                    [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, zorder=6))


def check(x, y, color, scale=1):
    line([(x-.07*scale, y), (x-.015*scale, y-.062*scale),
          (x+.10*scale, y+.075*scale)], color, lw=1.6, z=9)


def sheet(cx, cy, color, edge, *, scale=1, accepted=False, fill="white"):
    w, h = .54*scale, .70*scale
    x, y = cx-w/2, cy-h/2
    fold = .13*scale
    outline = [(x,y), (x+w,y), (x+w,y+h-fold), (x+w-fold,y+h), (x,y+h)]
    ax.add_patch(Polygon(outline, closed=True, facecolor=fill, edgecolor=edge,
                         lw=1.15, joinstyle="round", zorder=4))
    line([(x+w-fold,y+h), (x+w-fold,y+h-fold), (x+w,y+h-fold)],
         edge, lw=.85)
    for j, length in enumerate((.30,.23,.28)):
        yy = y+h-(.24+j*.11)*scale
        line([(x+.10*scale,yy),(x+(.10+length)*scale,yy)], color, lw=1.1)
    if accepted:
        ax.add_patch(Circle((x+w,y+.08*scale), .137*scale, fc="white",
                            ec=color, lw=1.1, zorder=7))
        check(x+w-.014*scale, y+.079*scale, color, scale*.66)


def questions(cx, cy):
    for dx, dy in ((.20,.13),(.10,.063),(0,0)):
        card(cx-.43+dx, cy-.33+dy, .86, .66, "white", PURPLE_LINE, r=.07, z=4)
    for j in range(3):
        y = cy+.16-j*.16
        selected = j == 1
        ax.add_patch(Circle((cx-.26, y), .042, ec=PURPLE, lw=1.05,
                            fc=PURPLE if selected else "white", zorder=7))
        line([(cx-.14,y),(cx+(.25 if selected else .17),y)],
             PURPLE if selected else PURPLE_LINE, lw=1.4)


def agent(cx, cy):
    card(cx-.38, cy-.26, .76, .52, "white", BLUE_LINE, r=.12, z=4)
    line([(cx,cy+.27),(cx,cy+.36)],BLUE,lw=1.1)
    ax.add_patch(Circle((cx,cy+.402),.043,fc=BLUE,ec="none",zorder=6))
    card(cx-.27,cy-.14,.54,.28,SKY,BLUE_LINE,r=.11,lw=.8,z=5)
    for dx in (-.105,.105):
        ax.add_patch(Circle((cx+dx,cy),.036,fc=BLUE,ec="none",zorder=6))
    line([(cx-.43,cy-.075),(cx-.43,cy+.075)],BLUE,lw=2.0)
    line([(cx+.43,cy-.075),(cx+.43,cy+.075)],BLUE,lw=2.0)


def split_icon(cx, cy):
    ax.add_patch(Polygon([(cx-.22,cy),(cx,cy+.19),(cx+.22,cy),(cx,cy-.19)],
                         closed=True, ec=AMBER, fc="white", lw=1.15, zorder=4))
    line([(cx-.46,cy),(cx-.22,cy)], AMBER)
    arrow([(cx+.22,cy),(cx+.34,cy),(cx+.34,cy+.25),(cx+.52,cy+.25)], AMBER, scale=8)
    arrow([(cx+.22,cy),(cx+.34,cy),(cx+.34,cy-.25),(cx+.52,cy-.25)], AMBER, scale=8)


def person(cx, cy):
    ax.add_patch(Circle((cx,cy+.16),.143,fc="white",ec=GREEN,lw=1.25,zorder=6))
    outline = PlotPath([(cx-.24,cy-.20), (cx-.23,cy+.09), (cx+.23,cy+.09),
                        (cx+.24,cy-.20), (cx-.24,cy-.20)],
                       [PlotPath.MOVETO,PlotPath.CURVE4,PlotPath.CURVE4,
                        PlotPath.CURVE4,PlotPath.CLOSEPOLY])
    ax.add_patch(PathPatch(outline, fc="#CBE5D5", ec=GREEN, lw=1.15, zorder=6))


def stage(x, w, title, caption, fill, edge, color):
    y, h = 2.34, 1.67
    card(x, y, w, h, fill, edge)
    label(x+w/2, y+h-.27, title, 14.0, color, weight="semibold")
    line([(x+.18,y+h-.50),(x+w-.18,y+h-.50)], edge, lw=.75, z=4)
    label(x+w/2,y+.25,caption,11.0,MUTED)
    return x+w/2


# Inference is read left to right. The wait path returns at a later state.
label(.39, 5.43, "At inference", 19.5, weight="semibold", align="left", serif=True)
line([(.40,5.12),(15.59,5.12)], RULE, lw=1.0, z=1)
c = stage(.40, 2.47, "Fixed agent", "Draft action and questions", SKY, BLUE_LINE, BLUE)
agent(c-.45, 3.15)
arrow([(c+.04,3.15),(c+.24,3.15)],BLUE,lw=1.0,scale=8)
sheet(c+.52, 3.16, BLUE, BLUE_LINE, scale=.69)
c = stage(3.18, 2.48, "Question selector", "Choose a local question", LILAC, PURPLE_LINE, PURPLE)
questions(c-.08, 3.10)
c = stage(5.97, 2.21, "Ask gate", "Ask now or wait", CREAM, AMBER_LINE, AMBER)
split_icon(c-.045, 3.17)

x, w = 8.83, 3.19
card(x,2.34,w,1.67,MINT,GREEN_LINE)
label(x+w/2,3.74,"Human judgment",14.0,GREEN,weight="semibold")
line([(x+.18,3.51),(x+w-.18,3.51)],GREEN_LINE,lw=.75,z=4)
person(9.25, 2.99)
card(9.64, 2.66, 2.18, .67, "white", GREEN_LINE, r=.075, z=4)
label(9.78,3.135,"Is chronicity established?",10.2,align="left",weight="medium")
ax.add_patch(Circle((9.84,2.878),.066,fc=MINT,ec=GREEN,lw=1,zorder=6))
check(9.832,2.877,GREEN,.46)
label(9.97,2.878,"Evidence is insufficient",10.1,GREEN,align="left")

c = stage(12.39, 3.20, "Action selector", "Choose an action and execute", SKY, BLUE_LINE, BLUE)
sheet(c-.47, 3.15, BLUE, BLUE_LINE, scale=.73)
label(c-.03,3.17,"or",10.5,MUTED)
sheet(c+.40,3.15,GREEN,GREEN_LINE,scale=.73)
label(c-.47,2.77,"Original",8.7,BLUE)
label(c+.40,2.77,"Revised",8.7,GREEN)

for left,right in ((2.87,3.18),(5.66,5.97)):
    arrow([(left,3.13),(right,3.13)])
arrow([(8.18,3.13),(8.83,3.13)],GREEN)
label(8.50,3.33,"Ask",9.8,GREEN,weight="semibold")
arrow([(12.02,3.13),(12.39,3.13)],GREEN)

# A raised return path makes the alternative explicit without a second pipeline.
arrow([(7.08,4.01),(7.08,4.61),(1.64,4.61),(1.64,4.01)], AMBER, dashed=True)
label(4.36,4.86,"Wait: execute the draft and retain the option to ask",11.4,AMBER)
# GPT revises the draft after the response; the original remains a candidate.
label(12.21,2.09,"The reply informs a revised action.",10.9,GREEN,serif=True)

# Replay and updates are separate from the online decision path.
card(.40,.26,15.19,1.42,NEUTRAL,RULE,r=.11)
label(.65,1.39,"Expert Iteration",16.0,weight="semibold",align="left",serif=True)
label(.65,.89,"Learn from completed branches",10.9,MUTED,align="left")
line([(4.26,.52),(4.26,1.41)],RULE,lw=1.0,z=4)

# Two observed branches, followed by an external evaluation.
for y,name,col,fc,ec in ((1.23,"Ask",GREEN,MINT,GREEN_LINE),(.69,"Wait",AMBER,CREAM,AMBER_LINE)):
    card(4.58,y-.145,.83,.29,fc,ec,r=.065,z=4)
    label(4.995,y,name,10.3,col,weight="semibold")
    curve_arrow((5.43,y),((5.68,y),(5.66,.96)),(5.98,.96),col,lw=1.15)
label(6.16,.96,"Replay",11.6,INK,align="left",weight="semibold")
arrow([(7.18,.96),(7.57,.96)],MUTED,lw=1.20)
sheet(7.99,.96,BLUE,BLUE_LINE,scale=.67,accepted=True)
label(8.40,1.10,"Score outcomes",11.7,INK,align="left",weight="semibold")
label(8.40,.80,"Quality and cost",9.7,MUTED,align="left")
arrow([(10.07,.96),(10.45,.96)],MUTED,lw=1.20)
label(10.71,.96,"Update",11.7,INK,align="left",weight="semibold")
for x,s,col,fc,ec in ((11.90,"A",BLUE,SKY,BLUE_LINE),
                      (12.90,"Q",PURPLE,LILAC,PURPLE_LINE),
                      (13.90,"G",AMBER,CREAM,AMBER_LINE)):
    card(x,.73,.58,.46,fc,ec,r=.07,z=4)
    label(x+.29,.96,s,13.0,col,weight="semibold")
for x in (12.48,13.48):
    arrow([(x+.06,.96),(x+.34,.96)],MUTED,lw=1.10,scale=10)

root = Path(__file__).resolve().parents[1]
runs = Path(os.environ.get("HUMANCHOICE_RUN_DIR", root / "runs")).expanduser()
out = runs / "legacy_figures" / "humanchoice-schematic.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, metadata={"Software": "Matplotlib", "Title": "HumanChoice overview"})
plt.close(fig)
print(out.name)
