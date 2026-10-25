"""WP1: recover publication IDs for the cleaned 1,930 rows by matching to the Malik et al. source xlsx."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, re, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = ROOT + ""
raw = pd.read_excel(R + "analysis/raw/uhpc_dataset_v2.xlsx", sheet_name="UHPC Dataset ", header=None)
raw = raw.iloc[3:].reset_index(drop=True)
raw[[88, 89, 90, 91]] = raw[[88, 89, 90, 91]].ffill()  # merged cells: label only on first row of each paper block
num = lambda c: pd.to_numeric(raw[c], errors="coerce")
src = pd.DataFrame({
    "paper": raw[91].astype(str).str.extract(r"Ref-(\d+)")[0].astype(float),
    "cement": num(1), "sf": num(4), "fa": num(5), "water": num(36), "sp": num(38),
    "sand": num(21), "cs28": num(53), "fib": num(25), "temp": num(40),
    "country": raw[88], "year": num(90),
})
print("source rows", len(src), "papers", src.paper.nunique(), "cs28 non-null", src.cs28.notna().sum())
Xtr = pd.read_csv(R + "Data/X_train_cleaned.csv"); ytr = pd.read_csv(R + "Data/y_train_cleaned.csv")
Xte = pd.read_csv(R + "Data/X_test_cleaned.csv"); yte = pd.read_csv(R + "Data/y_test_cleaned.csv")

def key(df, y):
    return pd.DataFrame({
        "cement": df["Cement Amount  (kg/m³)"], "sf": df["Silica Fume (kg/m³)"],
        "fa": df["Flayash Amount   (kg/m³)"], "water": df["Water _Amount  _(kg/m³)"],
        "sp": df["Superplasticizer  Amount  (kg/m³)"], "sand": df["Sand Amount  (kg/m³)"],
        "cs28": y.iloc[:, 0].values})
clean = pd.concat([key(Xtr, ytr).assign(split="train", row=range(len(Xtr))),
                   key(Xte, yte).assign(split="test", row=range(len(Xte)))], ignore_index=True)
K = ["cement", "sf", "fa", "water", "sp", "sand", "cs28"]
def norm(d):
    d = d.copy()
    for c in K: d[c] = d[c].fillna(0).astype(float).round(1)
    return d
c2, s2 = norm(clean), norm(src)
s2["src_idx"] = range(len(s2))
c2["occ"] = c2.groupby(K).cumcount(); s2["occ"] = s2.groupby(K).cumcount()
m = c2.merge(s2[K + ["occ", "paper", "src_idx", "country", "year", "temp", "fib"]], on=K + ["occ"], how="left")
# fallback: tolerance matching (float rounding such as 362.85 -> 362.8/362.9) on still-unused source rows
used = set(m.src_idx.dropna().astype(int))
S = src[K].fillna(0).astype(float).values
Craw = clean[K].fillna(0).astype(float).values
tol = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3])
n_fb = 0
for i in m.index[m.paper.isna()]:
    d = np.abs(S - Craw[i])
    ok = (d <= tol).all(axis=1)
    cand = [j for j in np.where(ok)[0] if j not in used and not np.isnan(src.paper.iloc[j])]
    if cand:
        j = min(cand, key=lambda j: d[j].sum())
        used.add(j); n_fb += 1
        m.loc[i, ["paper", "src_idx", "country", "year"]] = [src.paper.iloc[j], j, src.country.iloc[j], src.year.iloc[j]]
m["match_tier"] = np.where(m.paper.notna(), "exact_or_tolerance", "")
# tier 3: cleaned values differ from the source row (unit/consistency edits); assign the paper of the nearest source row
# by mean relative distance over the 7 key fields, requiring >=4 of 7 fields within 1%
Sn = np.where(src[K].isna(), 0, src[K].values).astype(float)
n3 = n4 = 0
for i in m.index[m.paper.isna()]:
    rel = np.abs(Sn - Craw[i]) / (np.abs(Craw[i]) + np.abs(Sn) + 1e-6)
    near = (rel <= 0.005).sum(axis=1)
    valid = ~src.paper.isna().values
    score = np.where(valid, near - rel.mean(axis=1), -9)
    j = int(np.argmax(score))
    tier = "nearest_ge4of7" if near[j] >= 4 else "nearest_weak"
    n3 += near[j] >= 4; n4 += near[j] < 4
    m.loc[i, ["paper", "src_idx", "country", "year", "match_tier"]] = [src.paper.iloc[j], j, src.country.iloc[j], src.year.iloc[j], tier]
print("tier3 (>=4/7 fields):", n3, " weak:", n4)
print("fallback matches", n_fb)
print("matched", m.paper.notna().sum(), "of", len(m), "papers", m.paper.nunique())
print(m[m.paper.isna()][K].describe().T[["count", "min", "max"]])
m[["split", "row", "paper", "src_idx", "country", "year", "match_tier"]].to_csv(R + "Data/groups.csv", index=False)
src.to_csv(R + "analysis/data/source_rows.csv", index_label="src_idx")
