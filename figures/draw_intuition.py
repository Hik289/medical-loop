"""Render the paper-style HumanChoice overview. All content is conceptual."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})
INK = "#162638"
MUTED = "#4c5e70"
BLUE = "#245c87"
SKY = "#eaf3fa"
BORDER = "#8facc1"
TEAL = "#166865"
MINT = "#e6f3f0"
AMBER = "#86541b"
CREAM = "#fbf2df"
LINE = "#dce5ec"

fig, ax = plt.subplots(figsize=(14.7, 5.9), dpi=205)
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 14.7), ylim=(0, 5.9), aspect="equal")
ax.axis("off")


def text(x, y, s, fs=15, c=INK, *, ha="center", bold=False):
    ax.text(x, y, s, size=fs, color=c, ha=ha, va="center",
            fontweight="semibold" if bold else "normal", zorder=5)


def box(x, y, w, h, fill, edge=BORDER, radius=.12, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               ec=edge, fc=fill, lw=lw, zorder=2))


def path(points, c=BLUE, *, dashed=False, arrow=True, lw=1.6):
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot((x1, x2), (y1, y2), color=c, lw=lw,
                ls=(0, (4, 3)) if dashed else "-", solid_capstyle="round", zorder=3)
    (x1, y1), (x2, y2) = points[-2:]
    if arrow:
        ax.add_patch(FancyArrowPatch((x1,y1), (x2,y2), arrowstyle="-|>",
                                    mutation_scale=12, lw=lw, color=c,
                                    shrinkA=0, shrinkB=0,
                                    linestyle=(0, (4, 3)) if dashed else "-", zorder=3))
    else:
        ax.plot((x1,x2), (y1,y2), color=c, lw=lw, zorder=3)


def node(x, y, w, h, title, sub, fill=SKY, edge=BORDER, color=INK):
    box(x, y, w, h, fill, edge)
    text(x+w/2, y+h*.66, title, 15, color, bold=True)
    text(x+w/2, y+h*.31, sub, 11.5, MUTED)


def record(cx, cy):
    x, y = cx-.26, cy-.31
    ax.add_patch(FancyBboxPatch((x, y), .52, .62,
                               boxstyle="round,pad=0,rounding_size=.045",
                               fc="white", ec=BLUE, lw=1.25, zorder=4))
    for i, w in enumerate((.31, .26, .19)):
        yy=y+.43-i*.12
        ax.plot((x+.10,x+.10+w), (yy,yy), color=BLUE, lw=1.35, zorder=5)


# Side-by-side composition, fine blue rules, serif titles and bounded accents.
text(.45, 5.50, "(a)  Autonomous agent", 19, ha="left", bold=True)
text(4.62, 5.50, "(b)  HumanChoice: one local question", 19, ha="left", bold=True)
ax.plot([.45, 4.04], [5.19, 5.19], c=LINE, lw=1.1)
ax.plot([4.62, 14.24], [5.19, 5.19], c=LINE, lw=1.1)
ax.plot([4.32, 4.32], [1.51, 5.06], c=LINE, lw=1.1)

box(.78, 3.92, 2.93, .87, "white")
record(1.24, 4.35)
text(1.67, 4.47, "Task evidence", 15, ha="left", bold=True)
text(1.67, 4.17, "Current state", 11.5, MUTED, ha="left")
path([(2.245,3.92),(2.245,3.60)])
node(.78,2.72,2.93,.88,r"Base agent $B$",r"Proposes action $a_0$")
path([(2.245,2.72),(2.245,2.39)])
node(.78,1.52,2.93,.87,"Execute and finish",r"Use $a_0$ without a question","white")

# Main row.
node(4.62,3.92,2.10,.87,r"Base agent $B$","Action + question pool")
node(7.05,3.92,2.18,.87,r"Select question $Q$","One local choice")
node(9.57,3.92,2.00,.87,r"Ask gate $G$","Ask now or wait?")
node(12.18,3.92,2.06,.87,"Wait",r"Use $a_0$; can ask later","white",BORDER,BLUE)
for a,b in ((6.72,7.05),(9.23,9.57),(11.57,12.18)):
    path([(a,4.355),(b,4.355)])

# Lower row: the Ask branch consumes the human opportunity once.
path([(10.57,3.92),(10.57,3.59),(8.65,3.59),(8.65,3.22)],TEAL)
text(9.62,3.75,"Ask",11.5,TEAL,bold=True)
box(4.62,1.52,4.42,1.70,MINT,"#6baba2")
text(4.88,2.98,"Human selects one option",14.5,TEAL,ha="left",bold=True)
text(4.88,2.64,"Does the record establish chronicity?",12,ha="left")
for y,label,picked in ((2.31,"Yes",False),(2.02,"No",False),(1.73,"Insufficient evidence",True)):
    ax.add_patch(Circle((4.98,y), .067, ec=TEAL, fc="white",lw=1.0,zorder=4))
    if picked:
        ax.add_patch(Circle((4.98,y), .033, ec="none", fc=TEAL,zorder=5))
    text(5.17,y,label,11.1,TEAL if picked else INK,ha="left",bold=picked)

node(9.40,1.52,2.10,1.11,r"Base agent $B$",r"Proposes revision $a_1$",SKY)
node(11.88,1.52,2.36,1.11,r"Action selector $A$",r"Choose $a_0$ or $a_1$; finish",SKY)
path([(9.04,2.07),(9.40,2.07)],TEAL)
path([(11.50,2.07),(11.88,2.07)])
# Preserve a0 as an option after the human answers.
path([(4.62,4.35),(4.47,4.35),(4.47,1.32),(13.06,1.32),(13.06,1.52)],
     BLUE,dashed=True,lw=1.15)
text(11.86,1.22,r"retain $a_0$",9.8,BLUE)

# Training is visually separate from the runtime paths.
box(.45,.35,13.79,.78,"#f6f8fb",LINE,radius=.10,lw=1.0)
text(.72,.75,"Expert Iteration",14.7,ha="left",bold=True)
text(3.10,.75,"Replay matched branches",12.2,ha="left")
path([(5.44,.75),(5.84,.75)],MUTED,lw=1.25)
text(6.04,.75,"Score completed tasks",12.2,ha="left")
path([(8.24,.75),(8.64,.75)],MUTED,lw=1.25)
text(8.88,.75,r"Update $A\ \rightarrow\ Q\ \rightarrow\ G$",13,ha="left",bold=True)
text(13.93,.75,"Train only",10.6,MUTED,ha="right")

out = Path(__file__).resolve().parents[1] / "assets" / "humanchoice-intuition.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=205, metadata={"Software": "Matplotlib", "Title": "HumanChoice overview"})
plt.close(fig)
print(out.name)
