"""WP0: reproduce Tables 3-4 of the submitted manuscript from the shipped checkpoints (sanity check of the pipeline)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT + "legacy")
import numpy as np, pandas as pd, torch, joblib
from inverse_design import BayesianNN
from grnbnn_lib import metrics, ROOT
Xte = pd.read_csv(ROOT + "Data/X_test_cleaned.csv"); yte = pd.read_csv(ROOT + "Data/y_test_cleaned.csv").iloc[:, 0].values
sx = joblib.load(ROOT + "models/assets/scaler_x.joblib"); sy = joblib.load(ROOT + "models/assets/scaler_y.joblib")
x = torch.tensor(sx.transform(Xte.values), dtype=torch.float32)
Ts = [1.5023, 1.3088, 1.2664, 1.5586, 1.2668]  # Table 4 of submitted manuscript
rows, mus, sds, raw = [], [], [], []
for k in range(5):
    m = BayesianNN.load_from_checkpoint(ROOT + f"models/checkpoints/fold_{k+1}_best.ckpt", input_dim=67, map_location="cpu")
    m.train(); torch.manual_seed(42)
    with torch.no_grad(): o = [m(x) for _ in range(60)]
    mu = torch.stack([a[0].squeeze() for a in o]).numpy(); sg = torch.stack([a[1].squeeze() for a in o]).numpy()
    s = sy.scale_[0]; mu = mu * s + sy.mean_[0]; sg = sg * s
    mm = mu.mean(0); sd = np.sqrt((sg ** 2).mean(0) + mu.var(0))
    T = Ts[k]; mus.append(mm); sds.append(T * sd); raw.append(sd)
    r = metrics(yte, mm, mm - 1.96 * T * sd, mm + 1.96 * T * sd); r["fold"] = k + 1; rows.append(r)
mus, sds = np.array(mus), np.array(sds)
mu_e = mus.mean(0); var_e = (sds ** 2).mean(0) + mus.var(0); sd_e = np.sqrt(var_e)
ens = metrics(yte, mu_e, mu_e - 1.96 * sd_e, mu_e + 1.96 * sd_e); ens["fold"] = "ensemble"; rows.append(ens)
# uncalibrated (T=1) ensemble for reference
raw = np.array(raw); sd_r = np.sqrt((raw ** 2).mean(0) + mus.var(0))
rows.append({**metrics(yte, mu_e, mu_e - 1.96 * sd_r, mu_e + 1.96 * sd_r), "fold": "ensemble T=1"})
df = pd.DataFrame(rows).round(4); print(df.to_string())
df.to_csv(ROOT + "analysis/results/wp0_reproduction.csv", index=False)
