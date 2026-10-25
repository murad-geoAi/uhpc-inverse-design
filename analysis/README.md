# Analysis pipeline

Every number, table and figure in the manuscript comes from the scripts in `code/`. Their outputs are in `results/` (tables as csv/json/npy) and `figures/` (600 dpi PNG and vector PDF). Paths resolve from `code/paths.py`, so the scripts can be run from any working directory:

```bash
python analysis/code/<script>.py
```

Seeds are fixed throughout. Everything runs on a CPU. One outer fold of the nested cross-validation took about 4–6 minutes on a laptop CPU (see `results/log_group.txt` and `results/log_random.txt`).

## Steps in order

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 0 | `wp0_reproduce.py` | Reproduces the version-1 results (80/20 split) from `models/` | `results/wp0_reproduction.csv` (Table S8) |
| 1a | `wp1_match_groups.py` | Matches the 1,930 rows to the Mendeley source workbook and recovers each row's publication | `Data/groups.csv`, `data/source_rows.csv` |
| 1b | `wp1_canonical_encoding.py` | Rebuilds the categorical inputs with one canonical label table | `Data/X_all_canonical.csv`, `Data/categorical_mappings_v2.json`, `results/encoding_audit.csv` (Table S5) |
| 1c | `wp1_exclusion_missingness.py` | Lists the source records not in the final data (with duplicate flags) and computes feature summaries and imputation shares | `results/removed_records.csv`, `exclusion_log.csv`, `exclusion_distribution.csv`, `feature_summary.csv`, `imputation_share.csv` (Tables S1–S4) |
| 2 | `run_cv.py group` / `run_cv.py random` | Nested five-fold CV of the GRN-BNN and every benchmark. The outer test fold is never used for training, epoch selection or calibration | `results/oof_*_all.csv`, `hist_*_all.csv`, `info_*_all.json` |
| 3 | `analyze_cv.py group all` / `analyze_cv.py random all` | Pooled metrics, publication-cluster bootstrap, per-fold, conditional coverage, uncertainty decomposition, within-publication ranking | `results/summary_*.csv` (Tables 4–6, S9–S12) |
| 4 | `train_final.py` | Trains the deployment ensemble on all data (publication-disjoint inner folds) | `models_v2/`, `results/oof_final_members.csv` |
| 5 | `interpret.py` | SHAP on the network and grouped permutation importance | `results/shap_*.npy`, `shap_importance.csv`, `grouped_permutation.csv` |
| 6 | `run_design.py 10` | Inverse-design experiments (contexts A–C, three risk modes, 10 restarts) | `results/design_results.json`, `design_summary.csv` (Table 7) |
| 7 | `hparam_sensitivity.py`, `outlier_study.py`, `chem_sensitivity.py` | Sensitivity studies | `results/hparam_sensitivity.csv` (Table 9), `outlier_study.csv` (Table S6), `chem_sensitivity.csv` (Table S13) |
| 8 | `fig_data.py`, `fig_schematics.py`, `fig_results.py`, `fig_interp.py`, `fig_design.py`, `gui_screenshots.py` | Figures 1–17 | `figures/` |

Steps 1a–1c need the source workbook. Download *UHPC Dataset (Version-2).xlsx* from https://doi.org/10.17632/czb7ww5pkz.2 and save it as `analysis/raw/uhpc_dataset_v2.xlsx`. It is not redistributed here. All later steps use the files in `Data/` only.

## Shared modules

- `grnbnn_lib.py` defines the GRN-BNN network, Student-t training, temperature scaling, the exact Student-t mixture and the metrics (PICP, MPIW, interval score, NLL, CRPS). `inverse_design_v2.py` and the app import it.
- `fig_style.py` holds the common figure style (colour-blind-safe palette, fonts, export settings).
- `paths.py` gives the repository root.

## Notes

- The tracebacks at the top of `results/log_design.txt` and `results/log_streamlit.txt` come from joblib's CPU-core detection on Windows. They are harmless warnings; the runs completed (see the last lines of each log).
- Software versions used for the reported results: Python 3.12, PyTorch 2.10, PyTorch Lightning 2.6, NumPy 1.26, pandas 2.2, SciPy 1.13, scikit-learn 1.8, XGBoost 2.1, LightGBM 4.5, CatBoost 1.2, SHAP 0.46, Matplotlib 3.8, Streamlit 1.32. The XGBoost and LightGBM scikit-learn wrappers do not work with scikit-learn 1.8, so the scripts use the native training APIs.
