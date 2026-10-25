"""Shared library for the revision experiments: GRN-BNN, Student-t mixture predictive distribution,
temperature scaling, and proper scoring metrics. Architecture mirrors inverse_design.py::BayesianNN."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats, optimize

ROOT = ROOT + ""
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def load_all(canonical=True):
    """Return X (1930x67 DataFrame), y (MPa), groups (paper id), is_orig_test (bool) in train-then-test order."""
    Xtr = pd.read_csv(ROOT + "Data/X_train_cleaned.csv"); Xte = pd.read_csv(ROOT + "Data/X_test_cleaned.csv")  # only used for sizes/targets
    ytr = pd.read_csv(ROOT + "Data/y_train_cleaned.csv").iloc[:, 0].values
    yte = pd.read_csv(ROOT + "Data/y_test_cleaned.csv").iloc[:, 0].values
    g = pd.read_csv(ROOT + "Data/groups.csv")
    X = pd.read_csv(ROOT + "Data/X_all_canonical.csv") if canonical else pd.concat([Xtr, Xte], ignore_index=True)
    y = np.concatenate([ytr, yte]).astype(float)
    assert len(g) == len(X) and (g.split.values[: len(Xtr)] == "train").all()
    return X, y, g.paper.values.astype(int), np.r_[np.zeros(len(Xtr), bool), np.ones(len(Xte), bool)], g


# ---------------------------------------------------------------- model
class GRNBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.fc1, self.fc2, self.gate = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop, self.norm = nn.Dropout(p), nn.LayerNorm(d)

    def forward(self, x):
        h = self.fc2(self.drop(F.elu(self.fc1(x))))
        return self.norm(x + torch.sigmoid(self.gate(x)) * h)


class GRNBNN(nn.Module):
    def __init__(self, d_in, hidden=256, depth=4, dropout=0.15):
        super().__init__()
        self.inp, self.skip = nn.Linear(d_in, hidden), nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([GRNBlock(hidden, dropout) for _ in range(depth)])
        self.mu, self.ls = nn.Linear(hidden, 1), nn.Linear(hidden, 1)
        self.raw_nu = nn.Parameter(torch.tensor(3.9))  # nu = 2.1 + softplus(raw) ~ 6 at init (finite variance)

    def nu(self):
        return 2.1 + F.softplus(self.raw_nu)

    def forward(self, x):
        h = F.gelu(self.inp(x) + self.skip(x))
        for b in self.blocks:
            h = b(h)
        sigma = F.softplus(self.ls(h).clamp(-6.0, 3.0)) + 1e-6
        return self.mu(h).squeeze(-1), sigma.squeeze(-1)


class DropMLP(nn.Module):
    """Baseline: 128-64-32 ReLU MLP, dropout at every layer. mode='mse' -> mean only, 'gauss' -> Gaussian NLL head."""
    def __init__(self, d_in, p=0.1, mode="mse"):
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p))
        self.mu, self.ls = nn.Linear(32, 1), nn.Linear(32, 1)

    def nu(self):
        return None

    def forward(self, x):
        h = self.net(x)
        return self.mu(h).squeeze(-1), (F.softplus(self.ls(h)) + 1e-4).squeeze(-1)


def student_t_nll(y, mu, sigma, nu):
    z = (y - mu) / sigma
    return -(torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * np.pi) - torch.log(sigma)
             - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))


def loss_fn(model, kind, xb, yb):
    mu, sg = model(xb)
    if kind == "mse":
        return F.mse_loss(mu, yb)
    if kind == "gauss":
        return (0.5 * torch.log(2 * np.pi * sg ** 2) + 0.5 * ((yb - mu) / sg) ** 2).mean()
    nu = model.nu()
    return student_t_nll(yb, mu, sg, nu).mean() + 0.25 * F.huber_loss(mu, yb, delta=0.5)


def _val_loss(model, kind, xv, yv):
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model, kind, xv, yv))


def fit(Xtr, ytr, Xva=None, yva=None, kind="studentt", arch="grn", max_epochs=200, fixed_epochs=None, patience=20,
        lr=7e-4, wd=1e-4, batch=64, seed=0, hidden=256, depth=4, dropout=0.15, drop_mlp=0.1):
    """Train one network on standardized data. If fixed_epochs is None: early stopping on (Xva,yva) -> returns best epoch.
    Otherwise trains exactly fixed_epochs epochs and never looks at validation data."""
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    model = GRNBNN(d, hidden, depth, dropout) if arch == "grn" else DropMLP(d, drop_mlp, "mse" if kind == "mse" else "gauss")
    n_ep = fixed_epochs if fixed_epochs is not None else max_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(max_epochs, n_ep) if fixed_epochs is None else n_ep)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    if Xva is not None:
        xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    best, best_ep, best_state, wait, hist = 1e18, 0, None, 0, []
    n = len(xt)
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(n)
        tl = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            l = loss_fn(model, kind, xt[idx], yt[idx]); l.backward(); opt.step(); tl += float(l) * len(idx)
        sched.step()
        if fixed_epochs is None and Xva is not None:
            vl = _val_loss(model, kind, xv, yv); hist.append((ep + 1, tl / n, vl))
            if vl < best - 1e-6:
                best, best_ep, wait = vl, ep + 1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, hist


@torch.no_grad()
def mc_predict(model, X, n_pass=60, seed=123, dropout_on=True):
    """Return mu [S,N], sigma [S,N] (standardized target scale). Dropout stays on (MC dropout)."""
    model.train() if dropout_on else model.eval()
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    out = [model(x) for _ in range(n_pass)]
    return torch.stack([o[0] for o in out]).numpy(), torch.stack([o[1] for o in out]).numpy()


# ---------------------------------------------------------------- calibration
def fit_T_gauss(y, mu_s, sg_s):
    """Gaussian-NLL temperature (the original recipe): member-level total sd = sqrt(mean sigma^2 + var mu)."""
    mu = mu_s.mean(0); sd = np.sqrt((sg_s ** 2).mean(0) + mu_s.var(0))
    f = lambda T: np.mean(np.log(T * sd) + 0.5 * ((y - mu) / (T * sd)) ** 2)
    return float(optimize.minimize_scalar(f, bounds=(0.2, 6), method="bounded").x)


def fit_T_t(y, mu_s, sg_s, nu):
    """Student-t temperature: minimise NLL of the per-member MC mixture of t(mu_s, T*sigma_s, nu)."""
    def f(T):
        pdf = stats.t.pdf((y[None] - mu_s) / (T * sg_s), nu) / (T * sg_s)
        return -np.mean(np.log(pdf.mean(0) + 1e-300))
    return float(optimize.minimize_scalar(f, bounds=(0.2, 6), method="bounded").x)


# ---------------------------------------------------------------- mixture predictive distribution
class Mixture:
    """Equal-weight mixture of location-scale components. comps: mu [C,N], scale [C,N], df (None -> Gaussian)."""
    def __init__(self, mu, scale, df=None):
        self.mu, self.scale, self.df = mu, scale, df

    def cdf(self, x):  # x [N]
        z = (x[None] - self.mu) / self.scale
        return (stats.norm.cdf(z) if self.df is None else stats.t.cdf(z, self.df)).mean(0)

    def logpdf(self, y):
        z = (y[None] - self.mu) / self.scale
        pdf = (stats.norm.pdf(z) if self.df is None else stats.t.pdf(z, self.df)) / self.scale
        return np.log(pdf.mean(0) + 1e-300)

    def mean(self):
        return self.mu.mean(0)

    def var_decomp(self):
        s2 = self.scale ** 2 if self.df is None else self.scale ** 2 * self.df / (self.df - 2)
        return s2.mean(0), self.mu.var(0)  # aleatoric, epistemic

    def ppf(self, q, iters=50):
        m = self.mu.mean(0); s = np.sqrt(sum(self.var_decomp())) * 12 + 1e-6
        lo, hi = m - s, m + s
        for _ in range(iters):
            mid = (lo + hi) / 2
            below = self.cdf(mid) < q
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2

    def crps(self, y, n=400):
        m = self.mu.mean(0); s = np.sqrt(sum(self.var_decomp())) * 10 + 1e-6
        t = np.linspace(-1, 1, n)
        out = np.empty(len(y))
        for i in range(len(y)):
            xs = np.minimum(np.maximum(m[i] + t * s[i], -1e9), 1e9)
            xs = np.union1d(xs, [y[i]])
            Fx = (stats.norm.cdf((xs[None] - self.mu[:, i:i+1]) / self.scale[:, i:i+1]) if self.df is None
                  else stats.t.cdf((xs[None] - self.mu[:, i:i+1]) / self.scale[:, i:i+1], self.df)).mean(0)
            out[i] = np.trapz((Fx - (xs >= y[i])) ** 2, xs)
        return out


# ---------------------------------------------------------------- metrics
def interval_score(y, lo, hi, alpha=0.05):
    return (hi - lo) + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)


def metrics(y, mu, lo=None, hi=None, nll=None, crps=None, spread=None):
    e = y - mu
    r = dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
             R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))
    if lo is not None:
        r.update(PICP=float(np.mean((y >= lo) & (y <= hi))), MPIW=float(np.mean(hi - lo)),
                 IS=float(np.mean(interval_score(y, lo, hi))))
    if nll is not None: r["NLL"] = float(np.mean(nll))
    if crps is not None: r["CRPS"] = float(np.mean(crps))
    sp = spread if spread is not None else (None if lo is None else (hi - lo))
    if sp is not None and np.std(sp) > 0:
        r["Spearman_spread_err"] = float(stats.spearmanr(sp, np.abs(e))[0])
    return r


def cluster_bootstrap(y, mu, lo, hi, groups, fn_names=("RMSE", "R2", "PICP", "MPIW", "IS"), B=1000, seed=0):
    """Paper-cluster bootstrap 95% CIs (resample publications, not rows)."""
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    res = {k: [] for k in fn_names}
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        m = metrics(y[ii], mu[ii], lo[ii], hi[ii])
        for k in fn_names: res[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}
