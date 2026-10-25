"""WP1: exclusion log (258 removed records), missingness per feature, outlier flags."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = ROOT + ""
raw = pd.read_excel(R + "analysis/raw/uhpc_dataset_v2.xlsx", sheet_name="UHPC Dataset ", header=None).iloc[3:].reset_index(drop=True)
g = pd.read_csv(R + "Data/groups.csv")
kept = set(g.src_idx.astype(int))
N = pd.to_numeric
src = pd.DataFrame({"cement": N(raw[1], errors="coerce"), "sf": N(raw[4], errors="coerce"), "water": N(raw[36], errors="coerce"),
                    "sp": N(raw[38], errors="coerce"), "sand": N(raw[21], errors="coerce"), "cs28": N(raw[53], errors="coerce"),
                    "fa": N(raw[5], errors="coerce"), "limestone": N(raw[7], errors="coerce"), "quartz": N(raw[8], errors="coerce"),
                    "glass": N(raw[9], errors="coerce"), "rha": N(raw[10], errors="coerce"), "mk": N(raw[11], errors="coerce"),
                    "ggbfs": N(raw[12], errors="coerce"), "slag": N(raw[13], errors="coerce"), "filler": N(raw[19], errors="coerce"),
                    "fiber": N(raw[25], errors="coerce")})
src["kept"] = src.index.isin(kept)
binder_cols = ["cement", "sf", "fa", "limestone", "quartz", "glass", "rha", "mk", "ggbfs", "slag"]
src["binder"] = src[binder_cols].fillna(0).sum(axis=1)
src["wb"] = src.water / src.binder
print("source rows:", len(src), " kept:", src.kept.sum(), " removed:", (~src.kept).sum())
rem = src[~src.kept].copy()
def reason(r):
    if np.isnan(r.cs28): return "R1 no 28-day compressive strength reported"
    if r.cement != r.cement or r.water != r.water or r.water <= 0: return "R2 cement or water content not reported"
    if r.sand != r.sand: return "R3 sand content not reported"
    if r.sp != r.sp: return "R4 superplasticiser content not reported"
    if r.wb < 0.10 or r.wb > 0.40: return "R5 water/binder outside 0.10-0.40"
    return "R6 removed during EDA outlier treatment and duplicate removal (record-level log not archived)"
rem["reason"] = rem.apply(reason, axis=1)
log = rem.reason.value_counts().sort_index(); print(log)
log.rename("records").to_csv(R + "analysis/results/exclusion_log.csv")
# distribution before / after
kp = src[src.kept]
cmp = pd.DataFrame({"all_28d (n=%d)" % src.cs28.notna().sum(): src.cs28.dropna().describe(), "retained (n=%d)" % len(kp): kp.cs28.describe(),
                    "removed with 28d (n=%d)" % rem.cs28.notna().sum(): rem.cs28.dropna().describe()}).round(1)
print(cmp); cmp.to_csv(R + "analysis/results/exclusion_distribution.csv")
print("retained W/B range:", round(kp.wb.min(),3), round(kp.wb.max(),3), " binder range:", kp.binder.min(), kp.binder.max())
# missingness in the raw source for retained rows, and what the cleaned file contains (zeros/means = imputed)
X = pd.read_csv(R + "Data/X_all_canonical.csv")
m = []
for c in X.columns:
    v = X[c]; zero = float((v == 0).mean()); mode_share = float(v.round(6).value_counts(normalize=True).iloc[0]); nun = int(v.nunique())
    m.append(dict(feature=c, n_unique=nun, pct_zero=round(100 * zero, 1), pct_most_common_value=round(100 * mode_share, 1),
                  min=float(v.min()), median=float(v.median()), max=float(v.max())))
pd.DataFrame(m).to_csv(R + "analysis/results/feature_summary.csv", index=False); print(pd.DataFrame(m).head(70).to_string())

# exact duplicates: same composition, curing and 28-day strength as an earlier source record (columns 1-45 and 53)
dupkey = raw[list(range(1, 46)) + [53]].astype(str).apply("|".join, axis=1)
rem["exact_duplicate"] = dupkey.duplicated(keep="first").loc[rem.index].values
rem[["reason", "exact_duplicate"]].assign(paper=raw[91].ffill().astype(str).str.extract(r"Ref-(\d+)")[0]).to_csv(R + "analysis/results/removed_records.csv", index_label="source_row_index")
# imputation share for material chemistry: fraction of rows with the material present whose oxide values equal the database-mean placeholder
import json
mat = json.load(open(R + "Data/materials.json", encoding="utf-8"))
amt = {"Cement": "Cement Amount  (kg/m³)", "Silica Fume": "Silica Fume (kg/m³)", "Fly Ash": "Flayash Amount   (kg/m³)", "Limestone Powder": "Limestone Powder  (kg/m3)",
       "Quartz Powder": "Quartzpowder (kg/m3)", "Glass Powder": "Glass powder (kg/m3)", "Rice Husk Ash": "Rice husk ash (kg/m3)", "Metakaolin": "Metakaolin (kg/m³)",
       "GGBFS": "GGBFS  (kg/m³)", "Steel Slag": "Slag Amount   (kg/m³)"}
out = []
for k, col in amt.items():
    present = X[col] > 0
    props = [c for c in mat[k] if c in X.columns]
    if not present.any() or not props: continue
    imp = np.mean([np.isclose(X.loc[present, c], mat[k][c], atol=1e-3).mean() for c in props])
    out.append(dict(material=k, n_mixes_using=int(present.sum()), pct_chemistry_equal_to_database_mean=round(100 * imp, 1)))
imp = pd.DataFrame(out); print(imp.to_string()); imp.to_csv(R + "analysis/results/imputation_share.csv", index=False)
