"""R2-4: one-at-a-time sensitivity of the GRN-BNN configuration under leave-publication-out 5-fold CV (single member per fold,
early stopping on a group-held-out validation split inside each training fold)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from grnbnn_lib import *
X, y, groups, _, _ = load_all(); Xv = X.values.astype(float)
base = dict(hidden=256, depth=4, dropout=0.15, lr=7e-4, wd=1e-4)
grid = [("base", {})] + [(f"{k}={v}", {k: v}) for k, vs in dict(hidden=[64, 128, 512], depth=[2, 3, 6], dropout=[0.0, 0.05, 0.30],
                                                                  wd=[1e-3, 1e-2, 1e-1], lr=[2e-4, 2e-3]).items() for v in vs]
rows = []
for name, ch in grid:
    cfg = {**base, **ch}; pred = np.zeros(len(y)); T0 = []
    for fo, (dev, te) in enumerate(GroupKFold(5).split(Xv, y, groups), 1):
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=fo).split(dev, groups=groups[dev]))
        sx, sy = StandardScaler().fit(Xv[dev]), StandardScaler().fit(y[dev, None]); Z = sx.transform(Xv); ys = sy.transform(y[:, None])[:, 0]
        m, be, _ = fit(Z[dev[a]], ys[dev[a]], Z[dev[b]], ys[dev[b]], kind="studentt", seed=fo, hidden=cfg["hidden"], depth=cfg["depth"],
                       dropout=cfg["dropout"], lr=cfg["lr"], wd=cfg["wd"]); T0.append(be)
        mu, _ = mc_predict(m, Z[te], 30); pred[te] = mu.mean(0) * sy.scale_[0] + sy.mean_[0]
    r = metrics(y, pred); r.update(config=name, mean_best_epoch=float(np.mean(T0)), **{k: cfg[k] for k in cfg}); rows.append(r)
    print(name, round(r["RMSE"], 2), round(r["R2"], 3), flush=True)
    pd.DataFrame(rows).to_csv(ROOT + "analysis/results/hparam_sensitivity.csv", index=False)
