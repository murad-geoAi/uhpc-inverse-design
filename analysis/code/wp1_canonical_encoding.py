"""WP1: rebuild the categorical columns of the 67-feature matrix from the source labels with ONE canonical encoding.
Finding: in the submitted X_train/X_test CSVs the integer codes of Type of Filler, Type of Fiber, Synergetic Type of Fiber, Curing,
Specimen Size and Max Size were assigned separately for the two files (and differ from the label table in materials.json used by the app).
Outputs: Data/X_all_canonical.csv (1930 x 67, train rows then test rows, same row order as groups.csv),
         Data/categorical_mappings_v2.json, analysis/results/encoding_audit.csv"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import sys, re, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = ROOT + ""
raw = pd.read_excel(R + "analysis/raw/uhpc_dataset_v2.xlsx", sheet_name="UHPC Dataset ", header=None).iloc[3:].reset_index(drop=True)
g = pd.read_csv(R + "Data/groups.csv"); src = g.src_idx.astype(int).values
X = pd.concat([pd.read_csv(R + "Data/X_train_cleaned.csv"), pd.read_csv(R + "Data/X_test_cleaned.csv")], ignore_index=True)
cols = {"Type of Fiber": 24, "Synergetic Type of Fiber": 30, "Type of Filler": 20, "Curing": 39, "Specimen Size": 46}


def norm(s, kind):
    s = str(s).replace("_x0002_", "").strip().lower(); s = re.sub(r"\s+", " ", s)
    if s in ("nan", ""): return "nan"
    if kind in ("Type of Fiber", "Synergetic Type of Fiber"):
        s = s.replace("fibre", "fiber").replace("seel", "steel"); s = re.sub(r"fibers\b", "fiber", s); s = re.sub(r"fibres\b", "fiber", s)
        s = re.sub(r"\bsteel fibers?\b", "steel fiber", s); s = s.replace("straight steel fiber", "straight steel fiber"); s = re.sub(r"\bstraight steel$", "straight steel fiber", s)
        s = s.replace("polyethylene fiber", "pe fiber").replace("pe fiber", "pe fiber")
        if s in ("pe", "pe fiber"): s = "pe fiber"
    if kind == "Curing":
        s = s.replace("stnadrad", "standard").replace("normal curing", "standard curing").replace("standard normal curing", "standard curing").replace("standard water curing", "standard curing")
        s = s.replace("steam  curing", "steam curing").replace("hot  curing", "hot curing")
    if kind == "Specimen Size":
        s = s.replace("×", "x").replace(" x ", "x").replace("x", " x ").replace("  ", " ").strip()
        s = re.sub(r"\s*x\s*", " x ", s)
    return s


canon, audit = {}, []
Xn = X.copy()
for cat, j in cols.items():
    lab = raw.loc[src, j].map(lambda v: norm(v, cat)).values
    uniq = sorted(set(lab) - {"nan"}); mapping = {"nan": 0, **{l: i + 1 for i, l in enumerate(uniq)}}
    canon[cat] = mapping; new = np.array([mapping[l] for l in lab], float)
    for sp in ("train", "test"):
        m = g.split.values == sp
        audit.append(dict(column=cat, file=sp, rows=int(m.sum()), rows_whose_code_changes=int((X[cat].values[m] != new[m]).sum()), distinct_labels_after_harmonisation=len(mapping)))
    Xn[cat] = new
# Max Size: true numeric (mm) in the source; the submitted file stored it as scrambled label codes
ms = pd.to_numeric(raw.loc[src, 23], errors="coerce").fillna(0.0).values
for sp in ("train", "test"):
    m = g.split.values == sp; audit.append(dict(column="Max Size (mm)", file=sp, rows=int(m.sum()), rows_whose_code_changes=int((np.abs(X["Max Size (mm)"].values[m] - ms[m]) > 1e-6).sum()), distinct_labels_after_harmonisation=-1))
Xn["Max Size (mm)"] = ms
# fibre property lookups by canonical fibre code (mean length / diameter of mixes that use that fibre)
fp, sfp = {}, {}
for code in range(len(canon["Type of Fiber"])):
    m = (Xn["Type of Fiber"] == code) & (Xn["Amount / Quantity of Fiber"] > 0)
    fp[str(code)] = {"Length (mm)": float(Xn.loc[m, "Length (mm)"].mean()) if m.any() else 0.0, "Diameter (mm)": float(Xn.loc[m, "Diameter (mm)"].mean()) if m.any() else 0.0}
for code in range(len(canon["Synergetic Type of Fiber"])):
    m = (Xn["Synergetic Type of Fiber"] == code) & (Xn["Amount / Quantity of Fiber .1"] > 0)
    sfp[str(code)] = {"Length (mm).1": float(Xn.loc[m, "Length (mm).1"].mean()) if m.any() else 0.0, "Diameter (mm).1": float(Xn.loc[m, "Diameter (mm).1"].mean()) if m.any() else 0.0}
Xn.to_csv(R + "Data/X_all_canonical.csv", index=False)
json.dump(dict(Categorical_Mappings=canon, Fiber_Properties=fp, Syn_Fiber_Properties=sfp), open(R + "Data/categorical_mappings_v2.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
a = pd.DataFrame(audit); a.to_csv(R + "analysis/results/encoding_audit.csv", index=False); print(a.to_string(index=False))
print({k: len(v) for k, v in canon.items()}); print("zero-fibre rows with fibre code != 0:", int(((Xn["Amount / Quantity of Fiber"] == 0) & (Xn["Type of Fiber"] != 0)).sum()),
      "| fibre rows with code 0:", int(((Xn["Amount / Quantity of Fiber"] > 0) & (Xn["Type of Fiber"] == 0)).sum()))
