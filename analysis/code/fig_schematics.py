import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import fig_style as fs; fs.setup(); P = fs.PAL
def box(ax, x, y, w, h, text, fc, ec=None, fs_=7, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec or fc, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(ax, p, q, c="#374151"): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=c, lw=1.2, shrinkA=0, shrinkB=0))
# ---------------- Fig 3: workflow
fig, ax = plt.subplots(figsize=(7.6, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
box(ax, 0.05, 3.9, 2.45, 1.3, "Source database\n(Malik et al.)\n2,188 mixes, 168 papers\n→ 1,930 mixes, 161 papers\n67 features", "#DBEAFE", P["blue"])
box(ax, 3.0, 3.9, 2.5, 1.3, "Leave-publication-out\nnested cross-validation\n(held-out papers never used\nfor training, stopping\nor calibration)", "#FEF3C7", P["gold"])
box(ax, 6.0, 3.9, 1.95, 1.3, "GRN-BNN ensemble\n5 members × 60\nMC-dropout passes\nStudent-t likelihood", "#DCFCE7", P["green"])
box(ax, 8.3, 3.9, 1.65, 1.3, "Student-t\ntemperature\nscaling on\nout-of-paper\npredictions", "#FCE7F3", P["purple"])
for a, b in [(2.5, 3.0), (5.5, 6.0), (7.95, 8.3)]: arrow(ax, (a, 4.55), (b, 4.55))
box(ax, 5.0, 2.0, 4.95, 1.2, "Calibrated predictive distribution\n(exact Student-t mixture: mean, σ, 90/95% intervals,\nP(strength ≥ target), aleatoric / epistemic split)", "#E0F2FE", P["teal"])
arrow(ax, (9.1, 3.9), (9.1, 3.2)); arrow(ax, (7.0, 3.9), (7.0, 3.2))
box(ax, 0.05, 0.35, 4.6, 1.9, "Reliability-aware inverse design\nmin (μ−y*)² + β σ² + w·max(0, σ−0.15μ)²\ns.t. binder, W/B, SP/B, S/B, filler/B, mass bounds\nSLSQP · autograd gradients · fixed dropout masks\n20 random restarts", "#FFEDD5", P["orange"])
arrow(ax, (5.0, 2.5), (4.65, 2.0)); arrow(ax, (4.65, 0.9), (5.3, 0.9))
box(ax, 5.5, 0.35, 4.4, 1.0, "Recommended mixture + mix-specific confidence statement\nStreamlit tool (same calibrated predictor as the paper)", "#EDE9FE", P["purple"], 6.6, bold=False)
fs.save(fig, "fig03_workflow")
# ---------------- Fig 4: architecture
fig, ax = plt.subplots(figsize=(7.6, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
box(ax, 0.1, 2.0, 1.2, 1.2, "Input\nx ∈ ℝ⁶⁷\n(standardised)", "#DBEAFE", P["blue"], 8)
box(ax, 1.7, 2.0, 1.5, 1.2, "Linear 67→256\n+ skip linear\nGELU", "#FEF3C7", P["gold"], 8)
arrow(ax, (1.3, 2.6), (1.7, 2.6))
x0 = 3.7
for i in range(4):
    box(ax, x0 + 1.2 * i, 1.5, 0.85, 2.2, f"GRN\nblock {i+1}\n\nFC–ELU\nDropout 0.15\nFC\n× σ(gate)\n+ residual\nLayerNorm", "#DCFCE7", P["green"], 5.6)
    arrow(ax, (x0 + 1.2 * i - 0.35 if i else 3.2, 2.6), (x0 + 1.2 * i, 2.6))
box(ax, 8.65, 3.2, 1.3, 0.8, "Mean head\nμ(x)", "#FCE7F3", P["purple"], 7)
box(ax, 8.65, 2.2, 1.3, 0.8, "Scale head\nσ(x)", "#FCE7F3", P["purple"], 7)
box(ax, 8.55, 1.2, 1.35, 0.8, "df ν (learned,\nν > 2.1)", "#FCE7F3", P["purple"], 7)
for yy in (3.6, 2.6, 1.6): arrow(ax, (8.45, 2.6), (8.64, yy))
ax.text(5.0, 0.55, "Dropout stays active at inference (Monte Carlo dropout, 60 passes per member); 5 members trained on publication-disjoint folds.\nTraining loss: Student-t negative log-likelihood + 0.25 × Huber (δ = 0.5) on the mean.  No attention mechanism is used.",
        ha="center", va="center", fontsize=7.5, color="#374151")
fs.save(fig, "fig04_architecture")
