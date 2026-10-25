"""WP5: inverse-design experiments with the rebuilt engine. Outputs results/design_results.json and design_summary.csv.
Usage: python run_design.py [restarts]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch
from scipy import stats
from inverse_design_v2 import InverseDesignV2

RES = ROOT + "analysis/results/"
restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 20
S = InverseDesignV2()
MODES = {"Aggressive": 0.0, "Balanced": 0.5, "Conservative": 2.0}
CONTEXTS = {
    "A: ambient curing, straight steel fibre, silica fume": dict(binder="Cement", scms=["Silica Fume"], filler=None, fiber="straight steel fiber", curing="standard curing", specimen="100 x 100 x 100"),
    "B: steam curing, straight steel fibre, silica fume": dict(binder="Cement", scms=["Silica Fume"], filler=None, fiber="straight steel fiber", curing="steam curing", specimen="100 x 100 x 100"),
    "C (submitted case): carbon fibre, ballast powder, ambient curing": dict(binder="Cement", scms=["Silica Fume"], filler="balast powder", fiber="carbon fibres (cf)", curing="standard curing", specimen="100 x 100 x 100"),
}
TARGETS = {"A": [120, 150, 200], "B": [150, 200, 250], "C": [150]}
out, rows = [], []
for cname, c in CONTEXTS.items():
    key = cname[0]
    P = S.build(c["binder"], c["scms"], c["filler"], c["fiber"], c["curing"], c["specimen"])
    for tgt in TARGETS[key]:
        sols_by_mode = {}
        for mname, beta in MODES.items():
            t0 = time.time(); best, sols = S.optimize(P, float(tgt), beta, restarts=restarts, seed=0)
            rec = dict(context=cname, target=tgt, mode=mname, beta=beta, time_s=round(time.time() - t0, 1), n_restarts=restarts, n_feasible=int(sum(s["feasible"] for s in sols)))
            if best is None:
                rec.update(feasible=False); out.append(rec); rows.append(rec); print(cname[0], tgt, mname, "INFEASIBLE", flush=True); continue
            funs = np.array([s["fun"] for s in sols if s["feasible"]])
            sig_r, mu_r = [], []
            for s_ in sols:
                if s_["feasible"]:
                    m_, sg_ = S.ens.moments(S.vector(P, torch.tensor(s_["x"], dtype=torch.float32)), 30, seed=777); sig_r.append(float(sg_)); mu_r.append(float(m_))
            rec_extra = dict(sigma_restarts_min=min(sig_r), sigma_restarts_max=max(sig_r), sigma_restarts_median=float(np.median(sig_r)), mu_restarts_min=min(mu_r), mu_restarts_max=max(mu_r))
            rec.update(rec_extra); rec.update(feasible=True, n_distinct_optima=int(len(np.unique(np.round(funs, 1)))), best_obj=float(best["fun"]), worst_feasible_obj=float(funs.max()))
            rep = S.report(P, best["x"], float(tgt))
            # Monte-Carlo noise: re-evaluate the same mixture with 10 independent sets of dropout masks (60 passes each)
            mc = []
            for sd_ in range(10):
                mix = S.ens.distribution(rep["vector"], 60, 5000 + sd_); a_, e_ = mix.var_decomp(); mc.append((float(mix.mean()[0]), float(np.sqrt(a_ + e_)[0]), float(mix.ppf(0.025)[0]), float(mix.ppf(0.975)[0])))
            mc = np.array(mc); rep.pop("vector")
            rec.update(rep); rec.update(mu_mc_mean=float(mc[:, 0].mean()), mu_mc_sd=float(mc[:, 0].std(ddof=1)), sigma_mc_mean=float(mc[:, 1].mean()), sigma_mc_sd=float(mc[:, 1].std(ddof=1)),
                       ci95_width_mc=float((mc[:, 3] - mc[:, 2]).mean()), sigma_mc_values=mc[:, 1].tolist())
            sols_by_mode[mname] = rec; out.append(rec)
            print(cname[0], tgt, mname, "mu %.1f sigma %.2f  P(within) %.2f  support=%s n_same=%d  active=%s  [%ss]" % (rep["mu"], rep["sigma"], rep["prob_within_tol"], rep["within_support"], rep["n_training_mixes_same_categories"], rep["active_constraints"], rec["time_s"]), flush=True)
        if "Aggressive" in sols_by_mode and "Conservative" in sols_by_mode:  # paired comparison over the 10 MC mask sets
            a, b = np.array(sols_by_mode["Aggressive"]["sigma_mc_values"]), np.array(sols_by_mode["Conservative"]["sigma_mc_values"])
            d = a - b; t, p = stats.ttest_rel(a, b)
            sols_by_mode["Conservative"].update(delta_sigma_vs_aggressive=float(d.mean()), delta_sigma_sd=float(d.std(ddof=1)), delta_sigma_p=float(p),
                                                 delta_mu_vs_aggressive=float(sols_by_mode["Conservative"]["mu"] - sols_by_mode["Aggressive"]["mu"]))
    json.dump(out, open(RES + "design_results.json", "w"), indent=1, default=float)
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sigma_mc_values", "mix")} | {f"mix_{k}": v for k, v in r.get("mix", {}).items()} for r in out])
df.to_csv(RES + "design_summary.csv", index=False); print("done")
