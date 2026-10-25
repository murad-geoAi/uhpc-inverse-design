# UHPC Inverse Design — Calibrated Bayesian Neural Network Ensemble

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reliability-aware inverse mix design for ultra-high-performance concrete (UHPC). The framework has four parts:

- a five-member gated residual network Bayesian neural network (GRN-BNN) ensemble with Monte Carlo dropout, a Student-t likelihood and Student-t temperature scaling;
- leave-publication-out evaluation against deep ensembles, conformal prediction, quantile regression, a Gaussian process and gradient-boosted trees;
- a deterministic multi-start SLSQP optimiser with documented constraints;
- a Streamlit application that runs the same calibrated predictor.

The repository accompanies the manuscript *“Probabilistic Inverse Design of Ultra High Performance Concrete Using a Calibrated Bayesian Neural Network Ensemble”* (Srabony, Murad and Islam; under review).

> **Read this first: what the model can and cannot do.** Evaluated across publications (leave-publication-out), the ensemble reaches R² ≈ 0.42 (RMSE ≈ 28 MPa), and its calibrated 95 % intervals are about ±77 MPa wide. Gradient-boosted trees are more accurate point predictors (R² ≈ 0.50–0.54). The tool is intended for screening and ranking candidate mixtures and for making uncertainty explicit. **None of the recommended mixtures has been cast and tested.**

## Repository structure

```
streamlit_app.py            Streamlit application (uses inverse_design_v2)
inverse_design_v2.py        calibrated predictor + deterministic multi-start SLSQP optimiser
models_v2/                  final ensemble: member_*.pt, calibration.json, scalers
Data/
  X_all_canonical.csv       1,930 x 67 feature matrix, canonical categorical encoding
  y_train_cleaned.csv,      28-day compressive strength (MPa); rows of X_all_canonical are
  y_test_cleaned.csv        the train rows followed by the test rows of the version-1 split
  groups.csv                publication identifier of every row (recovered from the source database)
  categorical_mappings_v2.json   label -> code tables used by the model and the app
  materials.json, realistic_constraints.json   material constants (oxides, specific gravity)
  X_train_cleaned.csv, X_test_cleaned.csv     version-1 matrices (only for reproducing version 1)
analysis/
  code/                     every script behind the paper (pipeline below)
  data/                     derived helper data (source-row index, empirical category codes)
  results/                  every table behind the paper (csv / json / npy)
  figures/                  all figures (600 dpi PNG + vector PDF)
  raw/                      place the Mendeley source workbook here (not redistributed)
legacy/                     version-1 engine and app, kept to reproduce the first results
models/                     version-1 checkpoints and scalers
```

## Installation

```bash
git clone https://github.com/murad-geoAi/uhpc-inverse-design.git
cd uhpc-inverse-design
pip install -r requirements.txt
```

Training and inference run on a CPU. On Windows with Anaconda, set `KMP_DUPLICATE_LIB_OK=TRUE` if PyTorch and NumPy load two OpenMP runtimes.

## Run the application

```bash
streamlit run streamlit_app.py
```

Defaults point to well-supported categories (straight steel fibre, standard curing, 100 mm cube, no filler). For each recommended mixture the app reports the predictive mean and standard deviation, the 90 % and 95 % intervals, P(|strength − target| ≤ 7.5 MPa), P(strength ≥ target) and the active constraints. It also warns when a selection is outside the data support or rare in the training data.

## Reproducing the results

All scripts resolve paths relative to the repository root, so they can be run from any directory. Random seeds are fixed. See [`analysis/README.md`](analysis/README.md) for the full pipeline, run times and the result file produced by each step.

```bash
# 0. Download "UHPC Dataset (Version-2).xlsx" from https://doi.org/10.17632/czb7ww5pkz.2
#    and save it as analysis/raw/uhpc_dataset_v2.xlsx (needed only for the data-preparation steps)
python analysis/code/wp0_reproduce.py              # version-1 results from the archived checkpoints
python analysis/code/wp1_match_groups.py           # publication identifiers -> Data/groups.csv
python analysis/code/wp1_canonical_encoding.py     # canonical categorical encoding
python analysis/code/wp1_exclusion_missingness.py  # removed-record list, missingness, imputation share
python analysis/code/run_cv.py group               # leave-publication-out nested CV, all models
python analysis/code/run_cv.py random              # random five-fold protocol
python analysis/code/analyze_cv.py group all       # pooled tables + publication-cluster bootstrap
python analysis/code/analyze_cv.py random all
python analysis/code/train_final.py                # final ensemble -> models_v2/
python analysis/code/interpret.py                  # SHAP on the network + grouped permutation
python analysis/code/run_design.py 10              # inverse-design experiments (10 restarts)
```

## Inverse-design problem

```
minimise   J(z) = (mu(z) - y*)^2 + beta * sigma(z)^2
subject to |mu(z) - y*| <= 7.5 MPa
           500 <= binder <= 1100 kg/m3        0.14 <= W/B <= 0.20      0.005 <= SP/B <= 0.03
           0.8 <= sand/B <= 1.5               filler/B <= 0.40         2200 <= total mass <= 2550 kg/m3
           each ingredient within the 1st-99th percentile of the database mixtures that use it
```

`mu` and `sigma` are the mean and standard deviation of the calibrated Student-t mixture. `beta` is 0, 0.5 or 2 for the aggressive, balanced and conservative modes. Dropout masks are fixed during the search, gradients come from automatic differentiation, and each problem is solved from several starting points.

## Version history

- **v2 (revision):** leave-publication-out evaluation and a canonical categorical encoding. Calibration uses a Student-t temperature with exact mixture quantiles, and benchmark models were added. The optimiser is deterministic with documented constraints, and the app runs the same calibrated predictor as the paper.
- **v1 (first submission):** a single random 80/20 split. Its engine (`legacy/inverse_design.py`), app (`legacy/streamlit_app_v1.py`) and checkpoints (`models/`) are kept unchanged. `analysis/code/wp0_reproduce.py` reproduces their results.

## Data, citation and license

Source database: U. J. Malik, D. Mohotti, H. Mo and C. K. Lee, *Global dataset of UHPC mix designs with supplementary cementitious materials, nano additives, and sustainable fillers*, Mendeley Data, V2 (2025), https://doi.org/10.17632/czb7ww5pkz.2. Please cite the source database and the accompanying article (see [`CITATION.cff`](CITATION.cff)).

Code and derived data in this repository are released under the MIT License.
