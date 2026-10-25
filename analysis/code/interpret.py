"""WP4: SHAP computed directly on the GRN-BNN ensemble (mean prediction) + grouped permutation importance on each member's
publication-held-out validation partition. Outputs CSV/NPY used by the figure script."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch, shap
from inverse_design_v2 import CalibratedEnsemble
from grnbnn_lib import load_all, ROOT
X, y, groups, _, _ = load_all(); Xv = X.values.astype(float); cols = list(X.columns)
ens = CalibratedEnsemble()
for net, _, _ in ens.members: net.eval()
def predict(A, members=None):
    x = (torch.tensor(np.asarray(A, float), dtype=torch.float32) - ens.xm) / ens.xs
    with torch.no_grad(): return np.mean([net(x)[0].numpy() for k, (net, _, _) in enumerate(ens.members) if members is None or k in members], 0) * ens.ys + ens.ym
G = {"Cement (amount + chemistry)": range(0, 5), "Silica fume": range(5, 10), "Fly ash": range(10, 15), "Limestone powder": range(15, 20), "Quartz powder": range(20, 25),
     "Glass powder": range(25, 29), "Rice husk ash": range(29, 34), "Metakaolin": range(34, 39), "GGBFS": range(39, 44), "Other slag": range(44, 49),
     "Filler (amount + type)": [49, 50], "Sand (amount, max size, d50)": [51, 52, 53], "Primary fibre": range(54, 58), "Secondary fibre": range(58, 62),
     "Water": [62], "Superplasticiser": [63], "Curing regime": [64], "Curing temperature": [65], "Specimen size": [66]}
# ---- grouped permutation importance on publication-held-out validation partitions of each member
rng = np.random.default_rng(0); rows = []
for k, m in enumerate(ens.cal["members"]):
    idx = np.array(m["val_rows_idx"]); Xb, yb = Xv[idx], y[idx]; base = np.sqrt(np.mean((yb - predict(Xb, [k])) ** 2))
    for gname, gi in G.items():
        gi = list(gi); d = []
        for _ in range(30):
            Xp = Xb.copy(); Xp[:, gi] = Xp[rng.permutation(len(idx))][:, gi]; d.append(np.sqrt(np.mean((yb - predict(Xp, [k])) ** 2)) - base)
        rows.append(dict(member=k + 1, group=gname, delta_rmse=float(np.mean(d))))
gp = pd.DataFrame(rows).groupby("group").delta_rmse.agg(["mean", "std"]).sort_values("mean", ascending=False); gp.to_csv(ROOT + "analysis/results/grouped_permutation.csv"); print(gp.round(2))
# ---- SHAP on the network itself (permutation explainer, background = 100 random mixtures, explained = 400 random mixtures)
bg = Xv[rng.choice(len(Xv), 100, replace=False)]; ex_i = rng.choice(len(Xv), 400, replace=False)
expl = shap.explainers.Permutation(predict, bg, max_evals=2 * 67 + 1)
sv = expl(Xv[ex_i]); np.save(ROOT + "analysis/results/shap_values.npy", sv.values); np.save(ROOT + "analysis/results/shap_rows.npy", ex_i)
imp = pd.Series(np.abs(sv.values).mean(0), index=cols).sort_values(ascending=False); imp.to_csv(ROOT + "analysis/results/shap_importance.csv"); print(imp.head(15).round(2))
# temperature dependence: mean SHAP by temperature band, and correlation with curing class
t = Xv[ex_i, 65]; s = sv.values[:, 65]
band = pd.cut(t, [-1, 25, 60, 100, 250], labels=["<=25C", "25-60C", "60-100C", ">100C"])
print(pd.DataFrame({"band": band, "shap_T": s}).groupby("band").shap_T.agg(["count", "mean"]).round(2))
