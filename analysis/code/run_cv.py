"""WP2/WP3: nested CV. Outer split = GroupKFold by publication ('group') or random KFold ('random').
Test fold is untouched by training, epoch selection and calibration (asserted).
Usage: python run_cv.py <group|random> [outer_fold_ids comma-separated] [--quick]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, time, warnings, pickle; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from grnbnn_lib import *

scenario = sys.argv[1]
only = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
QUICK = "--quick" in sys.argv
NP = 20 if QUICK else 60
X, y, groups, _, _ = load_all()
Xv = X.values.astype(float)
N = len(y)
if scenario == "group":
    outer = list(GroupKFold(5).split(Xv, y, groups))
else:
    outer = list(KFold(5, shuffle=True, random_state=42).split(Xv))
rows, hist_rows, info = [], [], []


def add(name, fold, idx, mu, lo, hi, sd=None, alea=None, epi=None, nll=None, crps=None, knn=None, pit=None):
    d = pd.DataFrame(dict(row=idx, outer=fold, scenario=scenario, method=name, y=y[idx], mu=mu, lo=lo, hi=hi))
    d["sd"] = sd if sd is not None else np.nan; d["alea"] = alea if alea is not None else np.nan
    d["epi"] = epi if epi is not None else np.nan; d["nll"] = nll if nll is not None else np.nan
    d["crps"] = crps if crps is not None else np.nan; d["knn"] = knn
    d["pit"] = pit if pit is not None else (stats.norm.cdf((y[idx] - mu) / sd) if sd is not None else np.nan)
    rows.append(d)


def inner_splits(dev, k=5):
    if scenario == "group":
        return [(dev[a], dev[b]) for a, b in GroupKFold(k).split(dev, groups=groups[dev])]
    return [(dev[a], dev[b]) for a, b in KFold(k, shuffle=True, random_state=7).split(dev)]


def gauss_interval(mu, sd, z=1.959964):
    return mu - z * sd, mu + z * sd


for fo, (dev, te) in enumerate(outer, start=1):
    if only and fo not in only:
        continue
    t0 = time.time()
    if scenario == "group":
        assert not set(groups[dev]) & set(groups[te]), "publication leakage between dev and test"
    assert not set(dev) & set(te)
    sx, sy = StandardScaler().fit(Xv[dev]), StandardScaler().fit(y[dev, None])
    Z = sx.transform(Xv); ys = sy.transform(y[:, None])[:, 0]; S = sy.scale_[0]; M = sy.mean_[0]
    nn_ = NearestNeighbors(n_neighbors=5).fit(Z[dev]); knn = nn_.kneighbors(Z[te])[0].mean(1)
    inn = inner_splits(dev)
    back = lambda m, s: (m * S + M, s * S)

    # ------------------------------------------------ GRN-BNN stage 1: epoch count from inner CV early stopping
    best_eps = []
    for k, (a, b) in enumerate(inn):
        _, be, h = fit(Z[a], ys[a], Z[b], ys[b], kind="studentt", seed=100 * fo + k)
        best_eps.append(be); hist_rows += [dict(outer=fo, inner=k, epoch=e, train=t, val=v) for e, t, v in h]
    n_ep = int(max(10, np.median(best_eps)))
    # ------------------------------------------------ stage 2: fixed epochs, OOF calibration, test prediction
    comps_mu, comps_sc_raw, nus, Tg, Tt, oof = [], [], [], [], [], []
    oof_mu = np.full(len(dev), np.nan); oof_sd = np.full(len(dev), np.nan); pos = {r: i for i, r in enumerate(dev)}
    member_te = []
    for k, (a, b) in enumerate(inn):
        m, _, _ = fit(Z[a], ys[a], kind="studentt", fixed_epochs=n_ep, seed=100 * fo + 10 + k)
        mv, sv = mc_predict(m, Z[b], NP, seed=k); nu = float(m.nu())
        Tg.append(fit_T_gauss(ys[b], mv, sv)); Tt.append(fit_T_t(ys[b], mv, sv, nu)); nus.append(nu)
        mu_b = mv.mean(0); sd_b = np.sqrt((sv ** 2).mean(0) + mv.var(0))
        for r, mm, ss in zip(b, mu_b, sd_b): oof_mu[pos[r]] = mm; oof_sd[pos[r]] = ss
        mt, st = mc_predict(m, Z[te], NP, seed=500 + k)
        member_te.append((mt, st, nu))
    assert not np.isnan(oof_mu).any()
    # (a) uncalibrated ensemble (T=1), (b) original Gaussian recipe, (c) Student-t mixture recipe
    def gauss_recipe(Ts):
        mus = np.array([mt.mean(0) for mt, st, nu in member_te])
        var = np.array([(T ** 2) * ((st ** 2).mean(0) + mt.var(0)) for T, (mt, st, nu) in zip(Ts, member_te)])
        mu = mus.mean(0); v = var.mean(0) + mus.var(0); return mu, np.sqrt(v)
    for name, Ts in [("GRN-BNN (T=1, uncalibrated)", [1] * 5), ("GRN-BNN (Gaussian TS, as submitted)", Tg)]:
        mu, sd = gauss_recipe(Ts); mu_o, sd_o = back(mu, sd); lo, hi = gauss_interval(mu_o, sd_o)
        nll = 0.5 * np.log(2 * np.pi * sd_o ** 2) + 0.5 * ((y[te] - mu_o) / sd_o) ** 2
        add(name, fo, te, mu_o, lo, hi, sd=sd_o, nll=nll, knn=knn)
    mu_c = np.concatenate([mt for mt, st, nu in member_te]); sc_c = np.concatenate([T * st for T, (mt, st, nu) in zip(Tt, member_te)])
    nu_e = float(np.mean(nus))
    mix = Mixture(mu_c * S + M, sc_c * S, nu_e)
    lo, hi = mix.ppf(0.025), mix.ppf(0.975); mu_o = mix.mean(); al, ep = mix.var_decomp()
    add("GRN-BNN (Student-t TS + mixture)", fo, te, mu_o, lo, hi, sd=np.sqrt(al + ep), alea=al, epi=ep,
        nll=-mix.logpdf(y[te]), crps=mix.crps(y[te]), knn=knn, pit=mix.cdf(y[te]))
    # (d) GRN-BNN + normalised split-conformal from OOF scores
    mu_oo, sd_oo = back(oof_mu, oof_sd); score = np.abs(y[dev] - mu_oo) / sd_oo
    q = np.quantile(score, min(1, np.ceil((len(dev) + 1) * 0.95) / len(dev)))
    mu_t, sd_t = gauss_recipe([1] * 5); mu_t, sd_t = back(mu_t, sd_t)
    add("GRN-BNN + normalised conformal", fo, te, mu_t, mu_t - q * sd_t, mu_t + q * sd_t, sd=sd_t, knn=knn)
    info.append(dict(outer=fo, n_epochs=n_ep, best_eps=best_eps, T_gauss=Tg, T_t=Tt, nu=nus, n_dev=len(dev), n_test=len(te),
                     n_dev_papers=int(len(set(groups[dev]))), n_test_papers=int(len(set(groups[te])))))

    # ------------------------------------------------ baselines: dropout-MLP (MSE), plain deep ensemble (Gauss NLL)
    m0, _, _ = fit(Z[inn[0][0]], ys[inn[0][0]], Z[inn[0][1]], ys[inn[0][1]], kind="mse", arch="mlp", lr=1e-3, wd=1e-5, seed=fo)
    mv, _ = mc_predict(m0, Z[te], NP, seed=1); mu_o, sd_o = back(mv.mean(0), mv.std(0))
    lo, hi = gauss_interval(mu_o, sd_o); add("Dropout-MLP baseline (MSE, as submitted)", fo, te, mu_o, lo, hi, sd=sd_o, knn=knn)
    ens_mu, ens_sd, Tm = [], [], []
    for k, (a, b) in enumerate(inn):
        m, _, _ = fit(Z[a], ys[a], Z[b], ys[b], kind="mse", arch="mlp", lr=1e-3, wd=1e-5, seed=fo * 7 + k)
        mvb, _ = mc_predict(m, Z[b], NP, seed=k); sdb = np.maximum(mvb.std(0), 1e-3)
        f = lambda T: np.mean(np.log(T * sdb) + 0.5 * ((ys[b] - mvb.mean(0)) / (T * sdb)) ** 2)
        T = float(optimize.minimize_scalar(f, bounds=(0.2, 40), method="bounded").x); Tm.append(T)
        mt, _ = mc_predict(m, Z[te], NP, seed=9 + k); ens_mu.append(mt.mean(0)); ens_sd.append(T * np.maximum(mt.std(0), 1e-3))
    ens_mu, ens_sd = np.array(ens_mu), np.array(ens_sd)
    mu = ens_mu.mean(0); sd = np.sqrt((ens_sd ** 2).mean(0) + ens_mu.var(0)); mu_o, sd_o = back(mu, sd)
    lo, hi = gauss_interval(mu_o, sd_o)
    add("Dropout-MLP + temperature scaling + ensemble", fo, te, mu_o, lo, hi, sd=sd_o, knn=knn)
    ens_mu, ens_sd, Tg2 = [], [], []
    for k, (a, b) in enumerate(inn):
        m, _, _ = fit(Z[a], ys[a], Z[b], ys[b], kind="gauss", arch="mlp", lr=1e-3, wd=1e-5, seed=fo * 11 + k, drop_mlp=0.0)
        mvb, svb = mc_predict(m, Z[b], 1, dropout_on=False); T = fit_T_gauss(ys[b], mvb, svb); Tg2.append(T)
        mt, st = mc_predict(m, Z[te], 1, dropout_on=False); ens_mu.append(mt[0]); ens_sd.append((st[0], T * st[0]))
    mus = np.array(ens_mu); sd_raw = np.sqrt((np.array([s[0] for s in ens_sd]) ** 2).mean(0) + mus.var(0))
    sd_cal = np.sqrt((np.array([s[1] for s in ens_sd]) ** 2).mean(0) + mus.var(0))
    for nm, sdv in [("Deep ensemble (plain MLP, Gaussian NLL)", sd_raw), ("Deep ensemble (plain MLP) + temperature scaling", sd_cal)]:
        mu_o, sd_o = back(mus.mean(0), sdv); lo, hi = gauss_interval(mu_o, sd_o); add(nm, fo, te, mu_o, lo, hi, sd=sd_o, knn=knn)

    # ------------------------------------------------ gradient boosting (point + conformal + quantile) and GP
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostRegressor
    class Nat:  # native-API adapters (the xgboost/lightgbm sklearn wrappers are incompatible with sklearn 1.8)
        def __init__(self, kind, **kw): self.kind, self.kw = kind, kw
        def fit(self, A, b):
            if self.kind == "xgb":
                self.m = xgb.train(dict(objective="reg:squarederror", eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, nthread=4, verbosity=0, seed=fo),
                                   xgb.DMatrix(A, label=b), num_boost_round=600)
            elif self.kind == "lgb":
                P = dict(objective="regression", learning_rate=0.03, num_leaves=31, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, num_threads=4, verbose=-1, seed=fo)
                P.update(self.kw.get("params", {})); self.m = lgb.train(P, lgb.Dataset(A, label=b), num_boost_round=self.kw.get("rounds", 800))
            else:
                self.m = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, verbose=0, thread_count=4, random_seed=fo).fit(A, b)
            return self
        def predict(self, A):
            return self.m.predict(xgb.DMatrix(A)) if self.kind == "xgb" else self.m.predict(A)
    mk = {"XGBoost": lambda: Nat("xgb"), "LightGBM": lambda: Nat("lgb"), "CatBoost": lambda: Nat("cat")}
    for nm, f in mk.items():
        oofp = np.full(len(dev), np.nan)
        for a, b in inn:
            oofp[[pos[r] for r in b]] = f().fit(Xv[a], y[a]).predict(Xv[b])
        res = np.abs(y[dev] - oofp); q = np.quantile(res, min(1, np.ceil((len(dev) + 1) * 0.95) / len(dev)))
        p = f().fit(Xv[dev], y[dev]).predict(Xv[te])
        add(nm + " + split-conformal", fo, te, p, p - q, p + q, sd=np.full(len(te), q / 1.96), knn=knn)
        add(nm + " (point only)", fo, te, p, np.full(len(te), np.nan), np.full(len(te), np.nan), knn=knn)
    # quantile LightGBM raw + CQR
    def qfit(a, al):
        return Nat("lgb", params=dict(objective="quantile", alpha=al, num_leaves=15), rounds=400).fit(Xv[a], y[a])
    sc = []
    for a, b in inn:
        ql, qh = qfit(a, 0.025).predict(Xv[b]), qfit(a, 0.975).predict(Xv[b]); sc.append(np.maximum(ql - y[b], y[b] - qh))
    sc = np.concatenate(sc); qhat = np.quantile(sc, min(1, np.ceil((len(dev) + 1) * 0.95) / len(dev)))
    ql, qh = qfit(dev, 0.025).predict(Xv[te]), qfit(dev, 0.975).predict(Xv[te]); pm = Nat("lgb").fit(Xv[dev], y[dev]).predict(Xv[te])
    add("LightGBM quantile regression (raw)", fo, te, pm, np.minimum(ql, qh), np.maximum(ql, qh), knn=knn)
    add("LightGBM conformalised quantile regression", fo, te, pm, np.minimum(ql, qh) - qhat, np.maximum(ql, qh) + qhat, knn=knn)
    # Gaussian process
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    gp = GPR(C(1.0) * Matern(length_scale=8.0, nu=2.5) + WhiteKernel(0.1), normalize_y=False, n_restarts_optimizer=0, random_state=0)
    gp.fit(Z[dev], ys[dev]); pm_, ps_ = gp.predict(Z[te], return_std=True); mu_o, sd_o = back(pm_, ps_)
    lo, hi = gauss_interval(mu_o, sd_o); add("Gaussian process (Matern-5/2)", fo, te, mu_o, lo, hi, sd=sd_o,
        nll=0.5 * np.log(2 * np.pi * sd_o ** 2) + 0.5 * ((y[te] - mu_o) / sd_o) ** 2, knn=knn)
    print(f"[{scenario}] outer {fo}/5 done in {time.time()-t0:.0f}s  (n_epochs={n_ep}, T_t={np.round(Tt,2)}, nu={np.round(nus,1)})", flush=True)
    tag = "all" if only is None else "_".join(map(str, only))
    pd.concat(rows).to_csv(f"{ROOT}analysis/results/oof_{scenario}{'_quick' if QUICK else ''}_{tag}.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(f"{ROOT}analysis/results/hist_{scenario}{'_quick' if QUICK else ''}_{tag}.csv", index=False)
    json.dump(info, open(f"{ROOT}analysis/results/info_{scenario}{'_quick' if QUICK else ''}_{tag}.json", "w"), indent=1)
