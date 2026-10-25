import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
import fig_style as fs; fs.setup(); P = fs.PAL
from inverse_design_v2 import InverseDesignV2
RES = ROOT + "analysis/results/"
R = [r for r in json.load(open(RES + "design_results.json")) if r.get("feasible")]
S = InverseDesignV2()
MODES = ["Aggressive", "Balanced", "Conservative"]; COL = {"Aggressive": P["red"], "Balanced": P["orange"], "Conservative": P["blue"]}
CTX = {"A": dict(binder="Cement", scms=["Silica Fume"], filler=None, fiber="straight steel fiber", curing="standard curing", specimen="100 x 100 x 100"),
       "B": dict(binder="Cement", scms=["Silica Fume"], filler=None, fiber="straight steel fiber", curing="steam curing", specimen="100 x 100 x 100")}
# ---------------- Fig 14: composition + predictive densities (context A, 150 MPa)
sel = {r["mode"]: r for r in R if r["context"][0] == "A" and r["target"] == 150}
c = CTX["A"]; Pb = S.build(c["binder"], c["scms"], c["filler"], c["fiber"], c["curing"], c["specimen"])
fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.6))
comps = ["Cement", "Silica Fume", "Sand", "Water", "Superplasticizer", "Fiber"]; w = 0.26; x_ = np.arange(len(comps))
for k, m in enumerate(MODES):
    if m not in sel: continue
    ax[0].bar(x_ + (k - 1) * w, [sel[m]["mix"].get(cn, 0) for cn in comps], w, color=COL[m], label=f"{m} (β = {sel[m]['beta']:g})")
ax[0].set_xticks(x_); ax[0].set_xticklabels(["Cement", "Silica fume", "Sand", "Water", "Super-\nplasticiser", "Steel\nfibre"], fontsize=8); ax[0].set_ylabel("Amount (kg/m³)"); ax[0].legend(fontsize=7.5)
xs = np.linspace(40, 260, 500)
for m in MODES:
    if m not in sel: continue
    r = sel[m]; xv = np.array([r["mix"].get(n, r["temperature"]) if n != "Temperature" else r["temperature"] for n in Pb["var"]])
    mix = S.ens.distribution(S.vector(Pb, torch.tensor(xv, dtype=torch.float32)).detach().numpy(), 60, 2024); cdf = mix.cdf(xs); pdf = np.gradient(cdf, xs)
    ax[1].plot(xs, pdf, color=COL[m], lw=1.8, label=f"{m}: μ = {r['mu']:.0f}, σ = {r['sigma']:.0f} MPa, P(window) = {r['prob_within_tol']:.2f}")
ax[1].axvspan(142.5, 157.5, color=P["light"], alpha=.7, label="Target window 150 ± 7.5 MPa"); ax[1].axvline(150, color="k", ls="--", lw=.8)
ax[1].set_xlabel("28-day compressive strength (MPa)"); ax[1].set_ylabel("Predictive density"); ax[1].legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=1, frameon=False); fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig14_design_mixes")
# ---------------- Fig 15: risk trade-off across targets
fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.5))
for a, ctx in zip(ax[:2], ["A", "B"]):
    tg = sorted({r["target"] for r in R if r["context"][0] == ctx})
    for k, m in enumerate(MODES):
        rr = [next((r for r in R if r["context"][0] == ctx and r["target"] == t and r["mode"] == m), None) for t in tg]
        xs_ = np.arange(len(tg)) + (k - 1) * 0.25
        a.bar(xs_, [r["sigma_mc_mean"] if r else np.nan for r in rr], 0.23, color=COL[m], label=m)
        if m == "Aggressive":
            lo = [r["sigma_restarts_min"] if r else np.nan for r in rr]; hi = [r["sigma_restarts_max"] if r else np.nan for r in rr]
            a.vlines(xs_, lo, hi, color="k", lw=1.2); a.scatter(xs_, lo, marker="_", color="k", s=40); a.scatter(xs_, hi, marker="_", color="k", s=40)
    a.set_xticks(range(len(tg))); a.set_xticklabels([f"{t}" for t in tg]); a.set_xlabel("Target strength (MPa)"); a.set_ylabel("Predictive σ of the solution (MPa)")
    a.set_title({"A": "Context A: ambient curing", "B": "Context B: steam curing"}[ctx], fontsize=9)
ax[0].legend(fontsize=7.5)
for m in MODES:
    xs_ = [(r["target"] + (MODES.index(m) - 1) * 3) for r in R if r["mode"] == m and r["context"][0] in "AB"]; ys_ = [r["mu"] - r["target"] for r in R if r["mode"] == m and r["context"][0] in "AB"]
    ax[2].scatter(xs_, ys_, color=COL[m], s=24, label=m)
ax[2].axhline(0, color="k", lw=.8); ax[2].axhspan(-7.5, 7.5, color=P["light"], alpha=.6); ax[2].set_xlabel("Target strength (MPa)"); ax[2].set_ylabel("Predicted mean − target (MPa)")
fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig15_design_tradeoff"); print("design figures done")
