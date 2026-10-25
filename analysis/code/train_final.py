"""Final (deployment) ensemble under the revised protocol: all 1,930 mixes, 5 group-wise inner folds (publication-disjoint),
epoch count from early stopping on held-out publications, fixed-epoch retraining, Student-t temperature from out-of-publication
predictions. Saves models_v2/ (state dicts), calibration.json, scalers."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings, os; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch, joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from grnbnn_lib import *
X, y, groups, _, _ = load_all(); Xv = X.values.astype(float)
OUT = ROOT + "models_v2/"; os.makedirs(OUT, exist_ok=True)
sx, sy = StandardScaler().fit(Xv), StandardScaler().fit(y[:, None]); Z = sx.transform(Xv); ys = sy.transform(y[:, None])[:, 0]
folds = list(GroupKFold(5).split(Z, ys, groups))
eps, hist = [], []
for k, (a, b) in enumerate(folds):
    _, be, h = fit(Z[a], ys[a], Z[b], ys[b], kind="studentt", seed=900 + k); eps.append(be)
    hist += [dict(inner=k, epoch=e, train=t, val=v) for e, t, v in h]
n_ep = int(max(10, np.median(eps))); print("best epochs", eps, "-> fixed", n_ep, flush=True)
cal = dict(n_epochs=n_ep, best_epochs=eps, members=[], x_columns=list(X.columns), y_mean=float(sy.mean_[0]), y_scale=float(sy.scale_[0]), mc_passes=60)
oof_mu = np.zeros(len(y)); oof_lo = np.zeros(len(y)); oof_hi = np.zeros(len(y)); oof_alea = np.zeros(len(y)); oof_epi = np.zeros(len(y))
for k, (a, b) in enumerate(folds):
    m, _, _ = fit(Z[a], ys[a], kind="studentt", fixed_epochs=n_ep, seed=950 + k)
    mv, sv = mc_predict(m, Z[b], 60, seed=k); nu = float(m.nu()); T = fit_T_t(ys[b], mv, sv, nu); Tg = fit_T_gauss(ys[b], mv, sv)
    mix = Mixture(mv * sy.scale_[0] + sy.mean_[0], T * sv * sy.scale_[0], nu)
    oof_mu[b] = mix.mean(); oof_lo[b] = mix.ppf(0.025); oof_hi[b] = mix.ppf(0.975); oof_alea[b], oof_epi[b] = mix.var_decomp()
    torch.save(m.state_dict(), OUT + f"member_{k+1}.pt")
    cal["members"].append(dict(file=f"member_{k+1}.pt", nu=nu, T_studentt=T, T_gauss=Tg, val_papers=int(len(set(groups[b]))), val_rows=int(len(b)),
                               val_rows_idx=[int(i) for i in b]))
    print(k + 1, "T_t", round(T, 2), "nu", round(nu, 2), flush=True)
cal["oof_metrics_member_level"] = metrics(y, oof_mu, oof_lo, oof_hi)
json.dump(cal, open(OUT + "calibration.json", "w"), indent=1)
joblib.dump(sx, OUT + "scaler_x.joblib"); joblib.dump(sy, OUT + "scaler_y.joblib")
pd.DataFrame(hist).to_csv(ROOT + "analysis/results/hist_final.csv", index=False)
pd.DataFrame(dict(row=np.arange(len(y)), y=y, mu=oof_mu, lo=oof_lo, hi=oof_hi, alea=oof_alea, epi=oof_epi, paper=groups)).to_csv(ROOT + "analysis/results/oof_final_members.csv", index=False)
print(cal["oof_metrics_member_level"])
