# UHPC Inverse Design — Streamlit App

Bayesian Neural Network–based inverse mix design tool for Ultra-High Performance Concrete (UHPC).

## Live App
Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

## Features
- Constraint-aware Bayesian optimization
- Posterior strength distribution with uncertainty quantification
- Mix composition profile and component ranking
- Academic-grade high-resolution chart exports

## Folder Structure
```
uhpc-streamlit-deploy/
├── streamlit_app.py          ← Main Streamlit application
├── inverse_design.py         ← BNN model + optimization engine
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Theme (light mode)
├── Data/
│   ├── materials.json        ← Material properties & categorical mappings
│   ├── X_train_cleaned.csv   ← Training features (for scaler fitting)
│   ├── y_train_cleaned.csv   ← Training targets (for scaler fitting)
│   └── realistic_constraints.json
└── models/
    └── checkpoints/
        ├── fold_1_best.ckpt
        ├── fold_2_best.ckpt
        ├── fold_3_best.ckpt
        ├── fold_4_best.ckpt
        └── fold_5_best.ckpt  ← BNN ensemble (5-fold cross-validated)
```

## Deploying to Streamlit Community Cloud
1. Push this folder as a **new GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and set **Main file path** to `streamlit_app.py`
4. Click **Deploy**

> **Note:** The `.ckpt` model files are ~10 MB each (~50 MB total).
> GitHub has a 100 MB file limit per file, so these are safe to push directly.
> If you exceed repo limits in the future, consider [Git LFS](https://git-lfs.github.com/).
