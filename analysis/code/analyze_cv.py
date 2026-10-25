"""Summaries from out-of-fold predictions: pooled metrics + paper-cluster bootstrap CIs, per-fold tables,
conditional coverage, aleatoric/epistemic split, within-publication ranking. Usage: python analyze_cv.py <group|random> [suffix]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy import stats
from grnbnn_lib import *

scenario = sys.argv[1]; suffix = sys.argv[2] if len(sys.argv) > 2 else "all"
RES = ROOT + "analysis/results/"
X, y, groups, _, _ = load_all()
d = pd.read_csv(f"{RES}oof_{scenario}_{suffix}.csv")
d["paper"] = groups[d.row.values]
mat = json.load(open(ROOT + "Data/materials.json", encoding="utf-8"))
inv = lambda cat: {v: k for k, v in mat["Categorical_Mappings"][cat].items()}
fib_names, cur_names = inv("Type of Fiber"), inv("Curing")


def fiber_class(r):
    if r["Amount / Quantity of Fiber"] <= 0: return "no fibre"
    n = fib_names.get(int(r["Type of Fiber"]), "")
    if "carbon" in n: return "carbon"
    if any(s in n for s in ["pe", "pp", "pva", "glass", "polyethylene"]) and "steel" not in n and "seel" not in n: return "synthetic"
    return "steel"


def curing_class(r):
    n = cur_names.get(int(r["Curing"]), "")
    return "heat/steam/autoclave" if any(s in n for s in ["steam", "hot", "heat", "autoclave", "oven", "warm", "90 oc"]) else "standard/ambient"


scm_cols = {"silica fume": "Silica Fume (kg/m³)", "fly ash": "Flayash Amount   (kg/m³)", "GGBFS": "GGBFS  (kg/m³)", "limestone": "Limestone Powder  (kg/m3)",
            "quartz powder": "Quartzpowder (kg/m3)", "glass powder": "Glass powder (kg/m3)", "rice husk ash": "Rice husk ash (kg/m3)",
            "metakaolin": "Metakaolin (kg/m³)", "steel slag": "Slag Amount   (kg/m³)"}
ctx = pd.DataFrame({"fibre": X.apply(fiber_class, axis=1), "curing": X.apply(curing_class, axis=1),
                    "scm": X[list(scm_cols.values())].idxmax(axis=1).map({v: k for k, v in scm_cols.items()})})
ctx["scm"] = np.where(X[list(scm_cols.values())].max(axis=1) <= 0, "none", ctx["scm"])
ctx["strength_bin"] = pd.cut(y, [0, 110, 130, 150, 170, 190, 400], labels=["<110", "110-130", "130-150", "150-170", "170-190", ">190"])
ctx.to_csv(RES + "context_labels.csv", index_label="row")
for c in ctx.columns: d[c] = ctx[c].values[d.row.values]

# ---------------- pooled metrics with bootstrap CIs
rows = []
for m, g in d.groupby("method", sort=False):
    ok = g.lo.notna().all()
    r = metrics(g.y.values, g.mu.values, g.lo.values if ok else None, g.hi.values if ok else None,
                nll=g.nll.values if g.nll.notna().all() else None, crps=g.crps.values if g.crps.notna().all() else None,
                spread=g.sd.values if g.sd.notna().all() else None)
    r["method"] = m
    if ok:
        ci = cluster_bootstrap(g.y.values, g.mu.values, g.lo.values, g.hi.values, g.paper.values, B=500)
        for k, (a, b) in ci.items(): r[k + "_lo"], r[k + "_hi"] = a, b
    rows.append(r)
pooled = pd.DataFrame(rows).set_index("method"); pooled.round(4).to_csv(f"{RES}summary_{scenario}_pooled.csv")
print(pooled[[c for c in ["RMSE", "MAE", "R2", "PICP", "PICP_lo", "PICP_hi", "MPIW", "IS", "NLL", "CRPS", "Spearman_spread_err"] if c in pooled]].round(3).to_string())

# ---------------- per outer fold
pf = d.groupby(["method", "outer"]).apply(lambda g: pd.Series(metrics(g.y.values, g.mu.values, *( (g.lo.values, g.hi.values) if g.lo.notna().all() else (None, None))))).reset_index()
pf.round(4).to_csv(f"{RES}summary_{scenario}_perfold.csv", index=False)

# ---------------- conditional coverage + width (key methods)
key = ["GRN-BNN (Student-t TS + mixture)", "GRN-BNN (Gaussian TS, as submitted)", "LightGBM conformalised quantile regression", "LightGBM + split-conformal"]
cc = []
for m in key:
    g = d[d.method == m]
    if g.empty: continue
    for c in ["strength_bin", "fibre", "curing", "scm"]:
        for lev, h in g.groupby(c, observed=True):
            if len(h) >= 10:
                cc.append(dict(method=m, factor=c, level=lev, n=len(h), PICP=np.mean((h.y >= h.lo) & (h.y <= h.hi)), MPIW=np.mean(h.hi - h.lo), RMSE=np.sqrt(np.mean((h.y - h.mu) ** 2))))
pd.DataFrame(cc).round(3).to_csv(f"{RES}summary_{scenario}_conditional.csv", index=False)

# ---------------- aleatoric / epistemic
g = d[d.method == "GRN-BNN (Student-t TS + mixture)"].copy()
if g.alea.notna().all():
    g["knn_bin"] = pd.qcut(g.knn, 4, labels=["Q1 (closest to training)", "Q2", "Q3", "Q4 (farthest)"])
    g["err"] = (g.y - g.mu).abs()
    out = g.groupby("knn_bin", observed=True).agg(n=("y", "size"), alea_sd=("alea", lambda v: np.sqrt(v.mean())), epi_sd=("epi", lambda v: np.sqrt(v.mean())),
                                                  MAE=("err", "mean"), PICP=("y", lambda v: 0)).drop(columns="PICP")
    out["epi_share_%"] = 100 * out.epi_sd ** 2 / (out.epi_sd ** 2 + out.alea_sd ** 2)
    out.round(2).to_csv(f"{RES}summary_{scenario}_alea_epi_by_distance.csv"); print(out.round(2))
    out2 = g.groupby("strength_bin", observed=True).agg(n=("y", "size"), alea_sd=("alea", lambda v: np.sqrt(v.mean())), epi_sd=("epi", lambda v: np.sqrt(v.mean())))
    out2.round(2).to_csv(f"{RES}summary_{scenario}_alea_epi_by_strength.csv")
    print("Spearman(knn dist, epistemic sd) =", round(stats.spearmanr(g.knn, np.sqrt(g.epi))[0], 3), " Spearman(knn dist, |err|) =", round(stats.spearmanr(g.knn, g.err)[0], 3))

# ---------------- within-publication ranking (can the model rank sibling mixes?)
rk = []
for m in ["GRN-BNN (Student-t TS + mixture)", "LightGBM + split-conformal", "XGBoost + split-conformal", "CatBoost + split-conformal", "Gaussian process (Matern-5/2)"]:
    g = d[d.method == m]
    r_ = [(len(h), stats.spearmanr(h.mu, h.y)[0]) for _, h in g.groupby("paper") if len(h) >= 5 and h.y.std() > 0]
    if r_:
        n_, rho = np.array(r_).T
        rk.append(dict(method=m, papers=len(r_), mean_rho=np.nanmean(rho), median_rho=np.nanmedian(rho), weighted_rho=np.nansum(n_ * rho) / n_.sum()))
pd.DataFrame(rk).round(3).to_csv(f"{RES}summary_{scenario}_within_paper_rank.csv", index=False); print(pd.DataFrame(rk).round(3).to_string())
