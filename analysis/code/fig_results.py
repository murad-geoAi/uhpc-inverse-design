import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
import fig_style as fs; fs.setup(); P = fs.PAL
RES = ROOT + "analysis/results/"
G = pd.read_csv(RES + "oof_group_all.csv"); R = pd.read_csv(RES + "oof_random_all.csv")
pg = pd.read_csv(RES + "summary_group_pooled.csv", index_col=0); pr = pd.read_csv(RES + "summary_random_pooled.csv", index_col=0)
GRN = "GRN-BNN (Student-t TS + mixture)"
NL = chr(10)
short = {GRN: "GRN-BNN" + NL + "(Student-t TS)", "GRN-BNN (Gaussian TS, as submitted)": "GRN-BNN" + NL + "(Gaussian TS)", "GRN-BNN (T=1, uncalibrated)": "GRN-BNN" + NL + "(uncalibrated)",
         "Dropout-MLP baseline (MSE, as submitted)": "Dropout-MLP" + NL + "(uncalibrated)", "Dropout-MLP + temperature scaling + ensemble": "Dropout-MLP" + NL + "+TS+ensemble",
         "Deep ensemble (plain MLP) + temperature scaling": "Deep ensemble" + NL + "+TS", "XGBoost + split-conformal": "XGBoost" + NL + "+conformal", "LightGBM + split-conformal": "LightGBM" + NL + "+conformal",
         "CatBoost + split-conformal": "CatBoost" + NL + "+conformal", "LightGBM conformalised quantile regression": "LightGBM" + NL + "CQR", "Gaussian process (Matern-5/2)": "Gaussian" + NL + "process"}
order = list(short)

# ------------------------------------------------ Fig 5: random vs publication-grouped evaluation
fig, ax = plt.subplots(1, 4, figsize=(11.5, 4.4), sharey=True)
y_ = np.arange(len(order))[::-1]
for a, (col, lab, ref) in zip(ax, [("RMSE", "RMSE (MPa)", None), ("PICP", "PICP at nominal 95%", 0.95), ("MPIW", "Mean 95% interval width (MPa)", None), ("IS", "Interval (Winkler) score, 95%", None)]):
    for off, (tab, c, nm) in zip([0.18, -0.18], [(pr, P["grey"], "Random 5-fold"), (pg, P["blue"], "Leave-publication-out 5-fold")]):
        v = [tab.loc[m, col] if m in tab.index else np.nan for m in order]
        a.barh(y_ + off, v, height=0.34, color=c, label=nm)
        if col == "PICP":
            lo = np.array([tab.loc[m, "PICP_lo"] for m in order]); hi = np.array([tab.loc[m, "PICP_hi"] for m in order]); vv = np.array(v)
            a.errorbar(vv, y_ + off, xerr=[vv - lo, hi - vv], fmt="none", ecolor="k", lw=.7, capsize=1.5)
    if ref: a.axvline(ref, color=P["red"], ls="--", lw=1)
    a.set_xlabel(lab)
ax[1].set_xlim(0.2, 1.02); ax[0].set_yticks(y_); ax[0].set_yticklabels([short[m] for m in order], fontsize=7.5); h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.55, 1.03))
fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig05_random_vs_group")

# ------------------------------------------------ Fig 6: learning curves
hg = pd.read_csv(RES + "hist_group_all.csv"); hr = pd.read_csv(RES + "hist_random_all.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for a, h, t in [(ax[0], hr, "Random inner folds"), (ax[1], hg, "Publication-disjoint inner folds")]:
    for col, c, lab in [("train", P["blue"], "Training loss"), ("val", P["orange"], "Validation loss (held-out)")]:
        s = h.groupby("epoch")[col].agg(["mean", "std", "count"]); s = s[s["count"] >= 5]
        a.plot(s.index, s["mean"], color=c, label=lab); a.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=.15)
    best = h.loc[h.groupby(["outer", "inner"]).val.idxmin()].epoch; a.axvline(best.median(), color=P["red"], ls=":", lw=1.2)
    a.set_title(t, fontsize=9); a.set_xlabel("Epoch"); a.set_xlim(0, 60)
    a.text(best.median() + 1, a.get_ylim()[1] * 0.97, "median best" + NL + "epoch = %d" % int(best.median()), color=P["red"], fontsize=7.5, va="top")
ax[0].set_ylabel("Loss (Student-t NLL + 0.25 Huber)"); ax[1].legend(fontsize=7.5); fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig06_learning_curves")

# ------------------------------------------------ Fig 7: parity plots
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.8))
for a, D, t in [(ax[0], R, "Random 5-fold CV"), (ax[1], G, "Leave-publication-out 5-fold CV")]:
    g = D[D.method == GRN].sample(500, random_state=1); cov = (g.y >= g.lo) & (g.y <= g.hi)
    a.errorbar(g.y[cov], g.mu[cov], yerr=[(g.mu - g.lo)[cov], (g.hi - g.mu)[cov]], fmt="none", ecolor=P["light"], lw=.6, zorder=1)
    a.errorbar(g.y[~cov], g.mu[~cov], yerr=[(g.mu - g.lo)[~cov], (g.hi - g.mu)[~cov]], fmt="none", ecolor="#F5B7B1", lw=.6, zorder=1)
    a.scatter(g.y[cov], g.mu[cov], s=6, color=P["blue"], zorder=2, label="95% interval covers"); a.scatter(g.y[~cov], g.mu[~cov], s=8, color=P["red"], zorder=3, label="not covered")
    a.plot([60, 320], [60, 320], "k--", lw=.8); a.set_xlim(60, 320); a.set_ylim(0, 340); a.set_xlabel("Measured strength (MPa)"); a.set_ylabel("Predicted strength (MPa)")
    d = D[D.method == GRN]; r2 = 1 - ((d.y - d.mu) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
    a.set_title(t + NL + "R2 = %.2f, RMSE = %.1f MPa, PICP = %.3f" % (r2, np.sqrt(((d.y - d.mu) ** 2).mean()), ((d.y >= d.lo) & (d.y <= d.hi)).mean()), fontsize=8.5)
ax[0].legend(fontsize=7, loc="upper left"); fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig07_parity")

# ------------------------------------------------ Fig 8: reliability + sigma-error
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.5))
lv = np.linspace(0.05, 0.99, 40)
for D, c, nm in [(R, P["grey"], "Random 5-fold"), (G, P["blue"], "Leave-publication-out")]:
    g = D[D.method == GRN]; ax[0].plot(lv, [np.mean(np.abs(g.pit - .5) <= p / 2) for p in lv], color=c, label=nm)
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("Nominal central-interval probability"); ax[0].set_ylabel("Empirical coverage"); ax[0].legend(fontsize=7.5)
for a, D, c, t in [(ax[1], R, P["grey"], "Random"), (ax[2], G, P["blue"], "Leave-publication-out")]:
    g = D[D.method == GRN].copy(); g["err"] = (g.y - g.mu).abs(); g["b"] = pd.qcut(g.sd, 10, labels=False)
    s = g.groupby("b").agg(sd=("sd", "mean"), err=("err", "mean"), e=("err", lambda v: v.std() / np.sqrt(len(v))))
    a.errorbar(s.sd, s.err, yerr=1.96 * s.e, fmt="o-", color=c, ms=3.5, lw=1)
    rho = stats.spearmanr(g.sd, g.err)[0]; a.set_title("%s" % t + NL + "Spearman rho(sigma, |error|) = %.2f" % rho, fontsize=8.5)
    a.set_xlabel("Predictive sigma, decile mean (MPa)"); a.set_ylabel("Mean |error| (MPa)")
fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig08_reliability_sigma")

# ------------------------------------------------ Fig 9/10: residual diagnostics (leave-publication-out)
g = G[G.method == GRN].copy(); g["res"] = g.y - g.mu; g["z"] = g.res / g.sd
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6)); ax = ax.ravel()
ax[0].hist(g.res, bins=45, color=P["blue"], alpha=.85, density=True); xs = np.linspace(-90, 90, 200); ax[0].plot(xs, stats.norm.pdf(xs, g.res.mean(), g.res.std()), color=P["red"]); ax[0].set_xlabel("Residual (MPa)"); ax[0].set_ylabel("Density")
ax[1].scatter(g.mu, g.res, s=4, alpha=.3, color=P["teal"], linewidths=0); ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("Predicted strength (MPa)"); ax[1].set_ylabel("Residual (MPa)")
ax[2].scatter(g.y, g.res, s=4, alpha=.3, color=P["orange"], linewidths=0); ax[2].axhline(0, color="k", lw=.8); ax[2].set_xlabel("Measured strength (MPa)"); ax[2].set_ylabel("Residual (MPa)")
q = stats.probplot(g.z, dist="norm"); ax[3].scatter(q[0][0], q[0][1], s=4, color=P["purple"]); ax[3].plot(q[0][0], q[0][0], "k--", lw=.8); ax[3].set_xlabel("Normal quantiles"); ax[3].set_ylabel("Standardised residual quantiles")
ax[3].text(.05, .92, "mean z = %.2f" % g.z.mean() + NL + "SD z = %.2f" % g.z.std(), transform=ax[3].transAxes, fontsize=8, va="top")
fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig09_residuals")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3)); m_ = (g.y + g.mu) / 2; bias, sdv = g.res.mean(), g.res.std()
ax[0].scatter(m_, g.res, s=4, alpha=.3, color=P["blue"], linewidths=0)
for v, c, l in [(bias, "k", "-"), (bias + 1.96 * sdv, P["red"], "--"), (bias - 1.96 * sdv, P["red"], "--")]: ax[0].axhline(v, color=c, ls=l, lw=1)
ax[0].set_xlabel("Mean of measured and predicted (MPa)"); ax[0].set_ylabel("Measured - predicted (MPa)"); ax[0].set_title("Bias %.1f MPa; limits +/-%.0f MPa" % (bias, 1.96 * sdv), fontsize=8.5)
ax[1].hist(g.z, bins=45, color=P["orange"], alpha=.85, density=True); zz = np.linspace(-4, 4, 200); ax[1].plot(zz, stats.norm.pdf(zz), color="k", lw=1, label="N(0,1)"); ax[1].set_xlim(-5, 5)
ax[1].set_xlabel("Standardised residual (y - mu)/sigma"); ax[1].legend(fontsize=7.5); fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig10_bland_altman")

# ------------------------------------------------ Fig 11: conditional behaviour
cond = pd.read_csv(RES + "summary_group_conditional.csv"); ae = pd.read_csv(RES + "summary_group_alea_epi_by_distance.csv", index_col=0); aer = pd.read_csv(RES + "summary_random_alea_epi_by_distance.csv", index_col=0)
fig, ax = plt.subplots(2, 3, figsize=(10.8, 6.0)); ax = ax.ravel()
c = cond[(cond.method == GRN) & (cond.factor == "strength_bin")]
ax[0].bar(c.level, c.RMSE, color=P["blue"]); ax[0].set_ylabel("RMSE (MPa)"); ax[0].set_xlabel("Measured strength bin (MPa)")
for i, n in enumerate(c.n): ax[0].text(i, 1, "n=%d" % n, ha="center", fontsize=6.5, color="white", rotation=90, va="bottom")
ax[1].bar(c.level, c.PICP, color=P["teal"]); ax[1].axhline(.95, color=P["red"], ls="--", lw=1); ax[1].set_ylim(.6, 1.02); ax[1].set_ylabel("PICP (95%)"); ax[1].set_xlabel("Measured strength bin (MPa)")
ax[2].bar(c.level, c.MPIW, color=P["orange"]); ax[2].set_ylabel("Mean 95% interval width (MPa)"); ax[2].set_xlabel("Measured strength bin (MPa)")
for a, f, t in [(ax[3], "fibre", "Fibre type"), (ax[4], "curing", "Curing regime")]:
    cc = cond[(cond.method == GRN) & (cond.factor == f)]; a.bar(cc.level, cc.PICP, color=P["purple"]); a.axhline(.95, color=P["red"], ls="--", lw=1); a.set_ylim(.6, 1.02); a.set_ylabel("PICP (95%)"); a.set_xlabel(t)
    for i, n in enumerate(cc.n): a.text(i, .62, "n=%d" % n, ha="center", fontsize=6.5, rotation=90, va="bottom", color="white")
w = .38; xx = np.arange(4)
for off, (t, c_, nm) in zip([-w / 2, w / 2], [(aer, P["grey"], "Random"), (ae, P["blue"], "Leave-publication-out")]):
    ax[5].bar(xx + off, t.epi_sd, w, color=c_, label=nm)
ax[5].set_xticks(xx); ax[5].set_xticklabels(["Q1" + NL + "nearest", "Q2", "Q3", "Q4" + NL + "farthest"]); ax[5].set_xlabel("Distance to training data (quartile)"); ax[5].set_ylabel("Epistemic sigma (MPa)"); ax[5].legend(fontsize=7)
for a in ax[:3]: a.tick_params(axis="x", labelsize=7.5)
fs.panel(ax, 10); fig.tight_layout(); fs.save(fig, "fig11_conditional")
print("figures 5-11 done")
