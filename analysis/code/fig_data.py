import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from scipy.stats import gaussian_kde
import fig_style as fs; fs.setup(); P = fs.PAL
from grnbnn_lib import load_all
X, y, grp, is_te, g = load_all()
binder_cols = ["Cement Amount  (kg/m³)", "Silica Fume (kg/m³)", "Flayash Amount   (kg/m³)", "Limestone Powder  (kg/m3)", "Quartzpowder (kg/m3)", "Glass powder (kg/m3)", "Rice husk ash (kg/m3)", "Metakaolin (kg/m³)", "GGBFS  (kg/m³)", "Slag Amount   (kg/m³)"]
binder = X[binder_cols].sum(axis=1); wb = X["Water _Amount  _(kg/m³)"] / binder
# ---------------- Figure 1
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.0)); ax = ax.ravel()
ax[0].hist(y, bins=36, color=P["blue"], alpha=.85, edgecolor="white", linewidth=.4); ax[0].axvline(150, color=P["red"], ls="--", lw=1.1)
ax[0].text(152, ax[0].get_ylim()[1] * .92, "150 MPa\n(UHPC threshold)", color=P["red"], fontsize=7.5, va="top")
ax[0].set_xlabel("28-day compressive strength (MPa)"); ax[0].set_ylabel("Number of mixtures")
m = (wb < 0.5) & (binder > 0); ax[1].scatter(wb[m], y[m], s=5, alpha=.35, color=P["teal"], linewidths=0)
ax[1].set_xlabel("Water-to-binder ratio"); ax[1].set_ylabel("28-day strength (MPa)"); ax[1].set_xlim(0.1, 0.45)
cnt = pd.Series(grp).value_counts().values
ax[2].bar(np.arange(len(cnt)), cnt, color=P["orange"], width=1.0); ax[2].set_xlabel("Publication (ranked by size)"); ax[2].set_ylabel("Mixtures per publication")
ax[2].text(0.97, 0.93, f"{len(cnt)} publications\nmedian {int(np.median(cnt))}, max {cnt.max()} mixtures", transform=ax[2].transAxes, ha="right", va="top", fontsize=8)
tr_pap = set(grp[~is_te]); share_rand = np.mean([p in tr_pap for p in grp[is_te]]) * 100
sh = []
for a, b in GroupKFold(5).split(X, y, grp): sh.append(np.mean([p in set(grp[a]) for p in grp[b]]) * 100)
bars = ax[3].bar(["Random 80/20\nsplit", "Leave-publication-\nout"], [share_rand, np.mean(sh)], color=[P["grey"], P["green"]], width=.55)
for b_, v in zip(bars, [share_rand, np.mean(sh)]): ax[3].text(b_.get_x() + b_.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax[3].set_ylabel("Held-out mixtures whose publication\nalso appears in training (%)"); ax[3].set_ylim(0, 115)
fs.panel(ax); fig.tight_layout(); fs.save(fig, "fig01_dataset")
# ---------------- Figure 2: train vs held-out distributions across leave-publication-out folds
feats = {"Binder content (kg/m³)": binder, "Silica fume (kg/m³)": X["Silica Fume (kg/m³)"], "Sand (kg/m³)": X["Sand Amount  (kg/m³)"],
         "Superplasticiser (kg/m³)": X["Superplasticizer  Amount  (kg/m³)"], "Curing temperature (°C)": X["Temperature (o C)"], "28-day strength (MPa)": pd.Series(y)}
fig, ax = plt.subplots(2, 3, figsize=(7.6, 4.6)); ax = ax.ravel()
for a_, (nm, v) in zip(ax, feats.items()):
    v = np.asarray(v, float); lo, hi = np.percentile(v, [0.5, 99.5]); xs = np.linspace(lo, hi, 200)
    tr, te = [], []
    for a, b in GroupKFold(5).split(X, y, grp):
        a_.plot(xs, gaussian_kde(v[b])(xs), color=P["orange"], alpha=.35, lw=.9)
    a_.plot(xs, gaussian_kde(v)(xs), color=P["blue"], lw=1.8, label="All data")
    a_.set_xlabel(nm); a_.set_yticks([]); a_.set_ylabel("Density")
ax[0].plot([], [], color=P["orange"], alpha=.6, lw=1, label="Held-out fold (5)"); ax[0].legend(loc="upper right", fontsize=7)
fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig02_distributions")
print("random-split share %.1f, group %.1f" % (share_rand, np.mean(sh)))
