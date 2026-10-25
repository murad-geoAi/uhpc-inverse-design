"""Does the placeholder chemistry (database-mean imputation) act as a publication fingerprint? Compare leave-publication-out accuracy with
(a) all 67 features, (b) no per-material chemistry/specific-gravity columns, (c) per-material chemistry replaced by binder-weighted CaO/SiO2/Al2O3."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, re, warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from grnbnn_lib import *

X, y, g, _, _ = load_all()
chem = [c for c in X.columns if re.match(r"^(CaO|SiO2|Al2O3|Sp\.Gr)_", c)]
amt = ["Cement Amount  (kg/m³)", "Silica Fume (kg/m³)", "Flayash Amount   (kg/m³)", "Limestone Powder  (kg/m3)", "Quartzpowder (kg/m3)", "Glass powder (kg/m3)", "Rice husk ash (kg/m3)", "Metakaolin (kg/m³)", "GGBFS  (kg/m³)", "Slag Amount   (kg/m³)"]
suf = ["C", "SF", "FA", "LP", "QP", "GP", "RH", "MK", "GGBFS", "S"]
B = X[amt].sum(axis=1).clip(lower=1)
Xb = X.drop(columns=chem).copy()
for ox in ["CaO", "SiO2", "Al2O3"]:
    Xb["binder_" + ox] = sum(X[a] * X[f"{ox}_{s}"] for a, s in zip(amt, suf) if f"{ox}_{s}" in X.columns) / B
variants = {"all 67 features (as used in the paper)": X.values.astype(float), "without per-material chemistry and specific gravity": X.drop(columns=chem).values.astype(float), "chemistry replaced by binder-weighted CaO, SiO2, Al2O3": Xb.values.astype(float)}
P = dict(objective="regression", learning_rate=0.03, num_leaves=31, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, num_threads=4, verbose=-1, seed=0)
rows = []
for name, A in variants.items():
    pc = np.zeros(len(y)); pl = np.zeros(len(y)); pg = np.zeros(len(y))
    for fo, (dev, te) in enumerate(GroupKFold(5).split(A, y, g), 1):
        pl[te] = lgb.train(P, lgb.Dataset(A[dev], label=y[dev]), 800).predict(A[te])
        pc[te] = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, verbose=0, thread_count=4, random_seed=0).fit(A[dev], y[dev]).predict(A[te])
        sx, sy = StandardScaler().fit(A[dev]), StandardScaler().fit(y[dev, None]); Z = sx.transform(A); ys = sy.transform(y[:, None])[:, 0]; mus = []
        for k in range(5):
            m, _, _ = fit(Z[dev], ys[dev], kind="studentt", fixed_epochs=10, seed=fo * 10 + k); mv, _ = mc_predict(m, Z[te], 20, seed=k); mus.append(mv.mean(0))
        pg[te] = np.mean(mus, 0) * sy.scale_[0] + sy.mean_[0]
    for mn, pr in [("GRN-BNN (5 members)", pg), ("LightGBM", pl), ("CatBoost", pc)]:
        r = metrics(y, pr); rows.append(dict(features=name, model=mn, n_features=A.shape[1], RMSE=r["RMSE"], R2=r["R2"]))
    print(name, flush=True)
df = pd.DataFrame(rows); df.round(3).to_csv(ROOT + "analysis/results/chem_sensitivity.csv", index=False); print(df.round(3).to_string())
