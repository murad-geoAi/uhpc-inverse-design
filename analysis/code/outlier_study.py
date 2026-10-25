"""R2-3: outlier handling study. Isolation Forest (fitted on each *training* fold only) flags ~5% of mixtures; models are retrained with and
without the flagged records and evaluated on ALL held-out records (leave-publication-out), so nothing is silently removed from evaluation."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from grnbnn_lib import *

X, y, g, _, _ = load_all(); Xv = X.values.astype(float); rows = []
P = dict(objective="regression", learning_rate=0.03, num_leaves=31, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, num_threads=4, verbose=-1, seed=0)
pred = {k: np.zeros(len(y)) for k in ["lgb_all", "lgb_clean", "grn_all", "grn_clean"]}; flagged_test = np.zeros(len(y), bool); n_flag = []
for fo, (dev, te) in enumerate(GroupKFold(5).split(Xv, y, g), 1):
    sx0 = StandardScaler().fit(np.c_[Xv[dev], y[dev]]); iso = IsolationForest(n_estimators=500, contamination=0.05, random_state=0).fit(sx0.transform(np.c_[Xv[dev], y[dev]]))
    keep = iso.predict(sx0.transform(np.c_[Xv[dev], y[dev]])) == 1; n_flag.append(int((~keep).sum()))
    flagged_test[te] = iso.predict(sx0.transform(np.c_[Xv[te], y[te]])) == -1
    for tag, idx in [("all", dev), ("clean", dev[keep])]:
        pred["lgb_" + tag][te] = lgb.train(P, lgb.Dataset(Xv[idx], label=y[idx]), 800).predict(Xv[te])
        sx, sy = StandardScaler().fit(Xv[idx]), StandardScaler().fit(y[idx, None]); Z = sx.transform(Xv); ys = sy.transform(y[:, None])[:, 0]; mus = []
        for k in range(5):
            m, _, _ = fit(Z[idx], ys[idx], kind="studentt", fixed_epochs=10, seed=fo * 10 + k); mv, _ = mc_predict(m, Z[te], 20, seed=k); mus.append(mv.mean(0))
        pred["grn_" + tag][te] = np.mean(mus, 0) * sy.scale_[0] + sy.mean_[0]
    print("fold", fo, "flagged in training", n_flag[-1], flush=True)
out = []
for k, v in pred.items():
    a = metrics(y, v); b = metrics(y[~flagged_test], v[~flagged_test]); out.append(dict(model=k, RMSE_all_test=a["RMSE"], R2_all_test=a["R2"], RMSE_unflagged_test=b["RMSE"], R2_unflagged_test=b["R2"]))
df = pd.DataFrame(out); df["flagged_total"] = int(flagged_test.sum()); df.round(3).to_csv(ROOT + "analysis/results/outlier_study.csv", index=False); print(df.round(3).to_string())
