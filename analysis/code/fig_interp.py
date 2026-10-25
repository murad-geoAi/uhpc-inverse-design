import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import fig_style as fs; fs.setup(); P = fs.PAL
RES = ROOT + "analysis/results/"
gp = pd.read_csv(RES + "grouped_permutation.csv").sort_values("mean")
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.barh(gp.group, gp["mean"], xerr=gp["std"], color=P["teal"], ecolor="k", error_kw=dict(lw=.8, capsize=2))
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Increase in out-of-publication RMSE when the feature group is permuted (MPa)")
fs.save(fig, "fig12_grouped_permutation")

X = pd.read_csv(ROOT + "Data/X_all_canonical.csv"); cols = list(X.columns)
sv = np.load(RES + "shap_values.npy"); rows = np.load(RES + "shap_rows.npy"); Xs = X.values[rows]
REN = {"Amount / Quantity of Fiber": "Fibre dosage (kg/m3)", "Max Size (mm)": "Sand max size (mm)", "Temperature (o C)": "Curing temperature (C)", "Quartzpowder (kg/m3)": "Quartz powder", "Flayash Amount   (kg/m³)": "Fly ash", "Water _Amount  _(kg/m³)": "Water", "Superplasticizer  Amount  (kg/m³)": "Superplasticiser", "Silica Fume (kg/m³)": "Silica fume", "Cement Amount  (kg/m³)": "Cement", "SiO2_GGBFS": "SiO2 of GGBFS", "Diameter (mm)": "Fibre diameter (mm)", "Length (mm)": "Fibre length (mm)", "SiO2_GP": "SiO2 of glass powder"}
clean = lambda c: REN.get(c) or c.replace("(kg/m³)", "").replace("(kg/m3)", "").replace("Amount", "").replace("_", " ").replace("  ", " ").strip()
imp = np.abs(sv).mean(0); top = np.argsort(imp)[::-1][:12]
fig = plt.figure(figsize=(10.4, 4.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
a = fig.add_subplot(gs[0]); rng = np.random.default_rng(0)
for k, j in enumerate(top[::-1]):
    v = Xs[:, j]; r = pd.Series(v).rank(pct=True).values; y_ = k + rng.uniform(-.3, .3, len(v))
    sc = a.scatter(sv[:, j], y_, c=r, cmap="coolwarm", s=5, linewidths=0, vmin=0, vmax=1)
a.set_yticks(range(len(top))); a.set_yticklabels([clean(cols[j]) for j in top[::-1]], fontsize=8); a.axvline(0, color="k", lw=.7)
a.set_xlabel("SHAP value: contribution to predicted strength (MPa)"); cb = fig.colorbar(sc, ax=a, pad=.01, fraction=.04); cb.set_label("Feature value (percentile)", fontsize=8)
b = fig.add_subplot(gs[1]); j = cols.index("Temperature (o C)"); T = Xs[:, j]
mat = None
b.scatter(T, sv[:, j], s=8, color=P["orange"], alpha=.7, linewidths=0)
for lo, hi, lab in [(-1, 25, "<=25 C"), (60, 100, "60-100 C"), (100, 250, ">100 C")]:
    m = (T > lo) & (T <= hi)
    if m.any(): b.hlines(sv[m, j].mean(), max(lo, 0), hi if hi < 250 else T.max(), color="k", lw=1.5); b.text((max(lo, 0) + min(hi, T.max())) / 2, sv[m, j].mean() + 2.5, "%+.1f MPa" % sv[m, j].mean(), ha="center", fontsize=8)
b.axhline(0, color="k", lw=.6); b.set_xlabel("Curing temperature (C)"); b.set_ylabel("SHAP value of curing temperature (MPa)")
fs.panel([a, b], 10); fig.tight_layout(); fs.save(fig, "fig13_shap")
print("ok")
