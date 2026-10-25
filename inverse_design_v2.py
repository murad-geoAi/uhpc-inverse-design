"""Reliability-aware inverse design, revised engine (v2).

Differences from inverse_design.py (v1):
  * the predictor is the *fully calibrated* ensemble: 5 members x 60 MC-dropout passes, Student-t temperature scaling,
    exact Student-t mixture predictive distribution (models_v2/calibration.json);
  * categorical inputs use ONE canonical label->code table (Data/categorical_mappings_v2.json) identical to the one used for training;
  * the optimiser objective is deterministic: dropout masks are fixed by a seed ("common random numbers"), so SLSQP sees a smooth
    function; exact gradients come from autograd and are passed as `jac`;
  * multi-start (R random restarts); the best feasible solution is returned;
  * the objective, every constraint and every bound are written out explicitly (OBJECTIVE_DOC, CONSTRAINTS, build()) and a
    bound-activity report is returned;
  * continuous variables are bounded by the training-data envelope (1st-99th percentile of the mixes that use that ingredient);
  * the confidence statement is the mix-specific predictive distribution of the recommended mixture, not a global PICP.
"""
import os, json, sys, re
import numpy as np, pandas as pd, torch, joblib
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "analysis", "code"))
from grnbnn_lib import GRNBNN, Mixture  # noqa: E402

TOL = 7.5  # MPa, half-width of the target window
OBJECTIVE_DOC = (
    "minimise J(z) = (mu(z) - y*)^2 + beta * sigma(z)^2  subject to  |mu(z) - y*| <= 7.5 MPa, the mixture constraints of CONSTRAINTS and the "
    "box bounds of the decision variables; mu and sigma are the mean and standard deviation of the calibrated Student-t mixture predictive distribution.")
CONSTRAINTS = {  # name: (lower, upper)
    "total binder (kg/m3)": (500.0, 1100.0), "water/binder": (0.14, 0.20), "SP/binder": (0.005, 0.03),
    "sand/binder": (0.8, 1.5), "filler/binder": (0.0, 0.40), "total mass (kg/m3)": (2200.0, 2550.0)}
CNAMES = ["binder>=500", "binder<=1100", "W/B>=0.14", "W/B<=0.20", "SP/B>=0.005", "SP/B<=0.03", "S/B>=0.8", "S/B<=1.5", "F/B<=0.40", "mass>=2200", "mass<=2550", "|mu-y*|<=7.5"]
BINDERS = ["Cement", "Silica Fume", "Fly Ash", "Limestone Powder", "Quartz Powder", "Glass Powder", "Rice Husk Ash", "Metakaolin", "GGBFS", "Steel Slag"]
COLMAP = {"Cement": "Cement Amount  (kg/m³)", "Silica Fume": "Silica Fume (kg/m³)", "Fly Ash": "Flayash Amount   (kg/m³)",
          "Limestone Powder": "Limestone Powder  (kg/m3)", "Quartz Powder": "Quartzpowder (kg/m3)", "Glass Powder": "Glass powder (kg/m3)",
          "Rice Husk Ash": "Rice husk ash (kg/m3)", "Metakaolin": "Metakaolin (kg/m³)", "GGBFS": "GGBFS  (kg/m³)", "Steel Slag": "Slag Amount   (kg/m³)"}


def normalize_label(kind, s):
    """Same harmonisation as analysis/code/wp1_canonical_encoding.py (used to build the training encoding)."""
    s = str(s).replace("_x0002_", "").strip().lower(); s = re.sub(r"\s+", " ", s)
    if s in ("nan", "", "none", "not specified"): return "nan"
    if kind in ("Type of Fiber", "Synergetic Type of Fiber"):
        s = s.replace("fibre", "fiber").replace("seel", "steel"); s = re.sub(r"fibers\b", "fiber", s); s = re.sub(r"fibres\b", "fiber", s)
        s = re.sub(r"\bsteel fibers?\b", "steel fiber", s); s = re.sub(r"\bstraight steel$", "straight steel fiber", s); s = s.replace("polyethylene fiber", "pe fiber")
        if s in ("pe", "pe fiber"): s = "pe fiber"
    if kind == "Curing":
        s = s.replace("stnadrad", "standard").replace("normal curing", "standard curing").replace("standard normal curing", "standard curing").replace("standard water curing", "standard curing")
    if kind == "Specimen Size":
        s = s.replace("×", "x"); s = re.sub(r"\s*x\s*", " x ", s)
    return s


class CalibratedEnsemble:
    def __init__(self, model_dir=None, device="cpu"):
        d = model_dir or os.path.join(BASE, "models_v2")
        self.cal = json.load(open(os.path.join(d, "calibration.json")))
        self.sx = joblib.load(os.path.join(d, "scaler_x.joblib")); self.sy = joblib.load(os.path.join(d, "scaler_y.joblib"))
        self.columns = self.cal["x_columns"]; self.members = []
        for m in self.cal["members"]:
            net = GRNBNN(len(self.columns)); net.load_state_dict(torch.load(os.path.join(d, m["file"]), map_location=device)); net.train()
            for p in net.parameters(): p.requires_grad_(False)
            self.members.append((net, m["T_studentt"], m["nu"]))
        self.xm = torch.tensor(self.sx.mean_, dtype=torch.float32); self.xs = torch.tensor(self.sx.scale_, dtype=torch.float32)
        self.ym, self.ys = float(self.cal["y_mean"]), float(self.cal["y_scale"])

    def components(self, x_raw, n_pass=60, seed=0):
        """Differentiable component means / scales (MPa) of the Student-t mixture; x_raw: tensor (67,) or (B,67)."""
        x = (x_raw.reshape(-1, len(self.columns)) - self.xm) / self.xs
        mus, scs, dfs = [], [], []
        for k, (net, T, nu) in enumerate(self.members):
            torch.manual_seed(seed + 1000 * k)
            mu, sg = net(x.repeat(n_pass, 1))  # independent dropout masks per row; fixed by the seed
            mus.append(mu.reshape(n_pass, -1) * self.ys + self.ym); scs.append(T * sg.reshape(n_pass, -1) * self.ys); dfs.append(nu)
        return torch.cat(mus, 0), torch.cat(scs, 0), float(np.mean(dfs))

    def moments(self, x_raw, n_pass=60, seed=0):
        mu, sc, df = self.components(x_raw, n_pass, seed)
        var = (sc ** 2 * df / (df - 2)).mean(0) + mu.var(0, unbiased=False)
        return mu.mean(0).squeeze(), torch.sqrt(var).squeeze()

    @torch.no_grad()
    def distribution(self, x_raw, n_pass=60, seed=0):
        mu, sc, df = self.components(torch.as_tensor(x_raw, dtype=torch.float32), n_pass, seed)
        return Mixture(mu.numpy(), sc.numpy(), df)


class InverseDesignV2:
    def __init__(self, model_dir=None, data_dir=None):
        from sklearn.neighbors import NearestNeighbors
        self.ens = CalibratedEnsemble(model_dir); dd = data_dir or os.path.join(BASE, "Data")
        self.mat = json.load(open(os.path.join(dd, "materials.json"), encoding="utf-8"))
        v2 = json.load(open(os.path.join(dd, "categorical_mappings_v2.json"), encoding="utf-8"))
        self.map = v2["Categorical_Mappings"]; self.fiber_props = v2["Fiber_Properties"]; self.syn_props = v2["Syn_Fiber_Properties"]
        self.mat = {**self.mat, "Categorical_Mappings": self.map, "Fiber_Properties": self.fiber_props, "Syn_Fiber_Properties": self.syn_props}
        self.cols = self.ens.columns; self.cidx = {c: i for i, c in enumerate(self.cols)}
        ref = pd.read_csv(os.path.join(dd, "X_all_canonical.csv")); self.ref = ref
        self.Zref = self.ens.sx.transform(ref[self.cols].values.astype(float))
        self._nn = NearestNeighbors(n_neighbors=6).fit(self.Zref)
        loo = self._nn.kneighbors(self.Zref)[0][:, 1:].mean(1)  # leave-one-out mean distance to the 5 nearest training mixes
        self.support_p95 = float(np.percentile(loo, 95)); self.support_median = float(np.median(loo))
        # data-envelope bounds (1st-99th percentile among mixes that use the ingredient)
        self.env = {}
        for n, c in list(COLMAP.items()) + [("Sand", "Sand Amount  (kg/m³)"), ("Filler", "Filler (kg/m³)"), ("Water", "Water _Amount  _(kg/m³)"),
                                            ("Superplasticizer", "Superplasticizer  Amount  (kg/m³)"), ("Fiber", "Amount / Quantity of Fiber")]:
            v = ref[c][ref[c] > (10 if n in COLMAP and n != "Cement" else 0)]
            self.env[n] = (float(np.percentile(v, 1)), float(np.percentile(v, 99))) if len(v) > 20 else (0.0, float(v.max()) if len(v) else 0.0)

    def code(self, cat, value):
        return float(self.map.get(cat, {}).get(normalize_label(cat, value), 0.0))

    # ------------------------------------------------------------------ problem set-up
    def build(self, binder_type, scm_types, filler_type, fiber_type, curing_type, specimen_size):
        scm_types = list(scm_types or []); var, lb, ub, x0 = [], [], [], []
        def add(n, lo, hi, g=None):
            e = self.env.get(n); lo2, hi2 = (max(lo, e[0]), min(hi, e[1])) if e else (lo, hi)
            if hi2 <= lo2: lo2, hi2 = lo, hi
            var.append(n); lb.append(lo2); ub.append(hi2); x0.append(g if g is not None and lo2 <= g <= hi2 else 0.5 * (lo2 + hi2))
        if binder_type == "Cement": add("Cement", 100.0, 900.0, 600.0)
        for s in scm_types: add(s, 10.0, 400.0, 100.0)
        add("Sand", 500.0, 1500.0, 900.0)
        has_filler = bool(filler_type) and normalize_label("Type of Filler", filler_type) != "nan"
        if has_filler: add("Filler", 0.0, 400.0, 100.0)
        add("Water", 100.0, 250.0, 160.0); add("Superplasticizer", 5.0, 50.0, 20.0)
        has_fiber = bool(fiber_type) and normalize_label("Type of Fiber", fiber_type) != "nan"
        if has_fiber: add("Fiber", 0.0, 500.0, 120.0)
        cc = self.code("Curing", curing_type); sel = self.ref[self.ref["Curing"] == cc]["Temperature (o C)"]
        tlo, thi = (float(sel.quantile(0.05)), float(sel.quantile(0.95))) if len(sel) >= 20 else (15.0, 40.0)
        if thi - tlo < 1.0: tlo, thi = max(tlo - 2, 0.0), thi + 2
        var.append("Temperature"); lb.append(tlo); ub.append(thi); x0.append(0.5 * (tlo + thi))
        base = np.zeros(len(self.cols)); A = np.zeros((len(var), len(self.cols)))
        for name in var:
            if name in COLMAP:
                A[var.index(name), self.cidx[COLMAP[name]]] = 1.0
                for p, v in self.mat.get(name, {}).items():
                    if p in self.cidx: base[self.cidx[p]] = v
        for k, v in self.mat.get("Sand", {}).items():
            if k in self.cidx: base[self.cidx[k]] = v
        for nm, col in [("Sand", "Sand Amount  (kg/m³)"), ("Filler", "Filler (kg/m³)"), ("Water", "Water _Amount  _(kg/m³)"),
                        ("Superplasticizer", "Superplasticizer  Amount  (kg/m³)"), ("Fiber", "Amount / Quantity of Fiber"), ("Temperature", "Temperature (o C)")]:
            if nm in var: A[var.index(nm), self.cidx[col]] = 1.0
        fc = self.code("Type of Fiber", fiber_type) if has_fiber else 0.0
        for c, v in [("Type of Filler", self.code("Type of Filler", filler_type) if has_filler else 0.0), ("Type of Fiber", fc), ("Curing", cc),
                     ("Specimen Size", self.code("Specimen Size", specimen_size)), ("Max Size (mm)", float(self.ref["Max Size (mm)"].median()))]:
            if c in self.cidx: base[self.cidx[c]] = v
        if has_fiber:
            fp = self.fiber_props.get(str(int(fc)), {})
            for c in ("Length (mm)", "Diameter (mm)"):
                if c in self.cidx: base[self.cidx[c]] = fp.get(c, 0.0)
        return dict(var=var, lb=np.array(lb), ub=np.array(ub), x0=np.array(x0), base=torch.tensor(base, dtype=torch.float32),
                    A=torch.tensor(A, dtype=torch.float32), has_fiber=has_fiber, has_filler=has_filler,
                    ctx=dict(binder=binder_type, scms=scm_types, filler=filler_type, fiber=fiber_type, curing=curing_type, specimen=specimen_size))

    @staticmethod
    def _ratios(P, x):
        v = {n: x[i] for i, n in enumerate(P["var"])}
        binder = sum(v[b] for b in BINDERS if b in v); mass = sum(x[i] for i, n in enumerate(P["var"]) if n != "Temperature")
        return binder, mass, v["Water"] / binder, v["Superplasticizer"] / binder, v["Sand"] / binder, v.get("Filler", torch.zeros(())) / binder

    def vector(self, P, x): return P["base"] + x @ P["A"]

    def geometric_constraints(self, P, x):  # all >= 0
        b, m, wb, sp, sa, fi = self._ratios(P, x)
        (b_lo, b_hi), (w_lo, w_hi), (s_lo, s_hi), (a_lo, a_hi), (_, f_hi), (m_lo, m_hi) = [CONSTRAINTS[k] for k in CONSTRAINTS]
        return torch.stack([b - b_lo, b_hi - b, wb - w_lo, w_hi - wb, sp - s_lo, s_hi - sp, sa - a_lo, a_hi - sa, f_hi - fi, m - m_lo, m_hi - m])

    # ------------------------------------------------------------------ optimisation
    def optimize(self, P, target, beta, restarts=20, n_pass_opt=30, seed=0, maxiter=200, tol=TOL):
        rng = np.random.default_rng(seed); best, sols = None, []
        def mom(xt): return self.ens.moments(self.vector(P, xt), n_pass_opt, seed=777)  # fixed masks => deterministic objective
        def f(x):
            xt = torch.tensor(x, dtype=torch.float32, requires_grad=True); mu, sg = mom(xt); v = (mu - target) ** 2 + beta * sg ** 2; v.backward()
            return float(v), xt.grad.double().numpy()
        def allc(xt):
            mu, _ = mom(xt); return torch.cat([self.geometric_constraints(P, xt), (tol ** 2 - (mu - target) ** 2).reshape(1)])
        def cf(x): return allc(torch.tensor(x, dtype=torch.float32)).detach().double().numpy()
        def cj(x): return torch.autograd.functional.jacobian(allc, torch.tensor(x, dtype=torch.float32)).double().numpy()
        for r in range(restarts):
            x0 = P["x0"] if r == 0 else rng.uniform(P["lb"], P["ub"])
            res = minimize(f, x0, jac=True, bounds=list(zip(P["lb"], P["ub"])), constraints=[{"type": "ineq", "fun": cf, "jac": cj}], method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-7})
            c = cf(res.x); scale = np.r_[np.ones(11), tol ** 2]; viol = float(max(0, -(c / scale).min())); feas = viol < 1e-3
            sols.append(dict(x=res.x, fun=float(res.fun), success=bool(res.success), feasible=feas, viol=viol, nit=int(res.nit)))
            if feas and (best is None or res.fun < best["fun"]): best = sols[-1]
        return best, sols

    # ------------------------------------------------------------------ reporting
    def report(self, P, x, target, n_pass=60, seed=2024, tol=TOL):
        xt = torch.tensor(x, dtype=torch.float32); vec = self.vector(P, xt).detach().numpy()
        mix = self.ens.distribution(vec, n_pass, seed); mu = float(mix.mean()[0]); al, ep = mix.var_decomp(); sd = float(np.sqrt(al + ep)[0])
        q = lambda p: float(mix.ppf(p)[0])
        cv = self.geometric_constraints(P, xt).numpy(); b, m, wb, sp, sa, fi = [float(a) for a in self._ratios(P, xt)]
        scales = [500.0, 1100.0, 0.14, 0.20, 0.005, 0.03, 0.8, 1.5, 0.4, 2200.0, 2550.0]  # constraint is "active" within 0.1% of its limit
        active = [n for n, c, s in zip(CNAMES[:11], cv, scales) if c < 1e-3 * s]
        at_bound = [n for n, xi, l, u in zip(P["var"], x, P["lb"], P["ub"]) if abs(xi - l) < 1e-3 * (1 + abs(l)) or abs(xi - u) < 1e-3 * (1 + abs(u))]
        if abs(mu - target) > tol - 1e-3: active.append(CNAMES[11])
        dist = float(self._nn.kneighbors(self.ens.sx.transform(vec.reshape(1, -1)), n_neighbors=5)[0].mean())
        ctx, ref = P["ctx"], self.ref; same = np.ones(len(ref), bool)
        for c, v in [("Type of Filler", self.code("Type of Filler", ctx["filler"]) if P["has_filler"] else 0.0), ("Type of Fiber", self.code("Type of Fiber", ctx["fiber"]) if P["has_fiber"] else 0.0),
                     ("Curing", self.code("Curing", ctx["curing"])), ("Specimen Size", self.code("Specimen Size", ctx["specimen"]))]:
            same &= (ref[c].values == v)
        same &= (ref["Amount / Quantity of Fiber"].values > 0) if P["has_fiber"] else (ref["Amount / Quantity of Fiber"].values == 0)
        return dict(mix={n: float(xi) for n, xi in zip(P["var"], x) if n != "Temperature"}, temperature=float(x[P["var"].index("Temperature")]),
                    mu=mu, sigma=sd, sigma_aleatoric=float(np.sqrt(al[0])), sigma_epistemic=float(np.sqrt(ep[0])), ci95=(q(0.025), q(0.975)), ci90=(q(0.05), q(0.95)),
                    prob_within_tol=float(mix.cdf(np.array([target + tol]))[0] - mix.cdf(np.array([target - tol]))[0]), prob_ge_target=float(1 - mix.cdf(np.array([target]))[0]),
                    binder=b, total_mass=m, wb=wb, sp_b=sp, sand_b=sa, filler_b=fi, active_constraints=active, variables_at_bounds=at_bound, vector=vec,
                    support_distance=dist, support_p95=self.support_p95, within_support=bool(dist <= self.support_p95), n_training_mixes_same_categories=int(same.sum()))


class InverseDesignSystem:
    """Drop-in replacement for inverse_design.InverseDesignSystem used by streamlit_app.py (same optimize_mix signature)."""
    def __init__(self, restarts=8):
        self.v2 = InverseDesignV2(); self.materials_data = self.v2.mat; self.restarts = restarts

    def optimize_mix(self, target_strength, binder_type, scm_types, filler_type, fiber_type, curing_type, specimen_size, beta=0.5,
                     reliability_threshold=None, return_diagnostics=False):
        P = self.v2.build(binder_type, scm_types, filler_type, fiber_type, curing_type, specimen_size)
        best, sols = self.v2.optimize(P, float(target_strength), float(beta), restarts=self.restarts)
        if best is None: raise RuntimeError("No feasible mixture meets the +/-7.5 MPa target window for these inputs; try different material selections or target strength.")
        r = self.v2.report(P, best["x"], float(target_strength))
        width = r["ci95"][1] - r["ci95"][0]; status = "Informative" if width <= 0.5 * r["mu"] else "Wide interval"
        if not r["within_support"]: status += " | outside data support"
        if r["n_training_mixes_same_categories"] < 10: status += " | sparse categories"
        mix_dist = self.v2.ens.distribution(r["vector"], 60, 2024); grid = np.linspace(r["mu"] - 5 * r["sigma"], r["mu"] + 5 * r["sigma"], 400)
        cdf = mix_dist.cdf(grid)
        diag = dict(optimization_success=bool(best["success"]), predicted_strength=r["mu"], uncertainty=r["sigma"], confidence_score=100.0 * r["prob_within_tol"],
                    reliability_status=status, ci95=r["ci95"], ci90=r["ci90"], prob_within_tolerance=r["prob_within_tol"], prob_ge_target=r["prob_ge_target"],
                    sigma_aleatoric=r["sigma_aleatoric"], sigma_epistemic=r["sigma_epistemic"], within_data_support=r["within_support"],
                    n_training_mixes_same_categories=r["n_training_mixes_same_categories"], active_constraints=r["active_constraints"], variables_at_bounds=r["variables_at_bounds"],
                    density_grid=[grid.tolist(), np.gradient(cdf, grid).tolist(), cdf.tolist()],
                    engineering_props={"Total Binder": r["binder"], "Total Mass": r["total_mass"], "W/B Ratio": r["wb"], "Temperature": r["temperature"]})
        out = (r["mix"], r["mu"], r["sigma"], diag)
        return out if return_diagnostics else out[:3]
