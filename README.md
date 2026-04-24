# UHPC Inverse Design — Bayesian Neural Network Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-red?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Constraint-aware inverse mix design for Ultra-High Performance Concrete (UHPC) powered by a Bayesian Neural Network ensemble with uncertainty quantification.**

---

## Overview

This repository implements an **inverse design system** for UHPC mix proportioning. Given a target compressive strength, the system uses a trained **Bayesian Neural Network (BNN)** ensemble to find the optimal mix design that satisfies real-world engineering constraints — while quantifying prediction uncertainty.

The interactive dashboard is built with **Streamlit** and provides:

- One-click mix optimization across multiple risk strategies
- Posterior strength distribution with 68 % / 95 % credible intervals
- Composition charts and component ranking
- Downloadable results

---

## Repository Structure

```
uhpc-inverse-design/
│
├── streamlit_app.py            # 🚀 Main dashboard (UI + visualization)
├── inverse_design.py           # 🧠 BNN model + SLSQP optimization engine
├── requirements.txt            # 📦 Python dependencies
├── .gitignore                  # 🙈 Git ignore rules
├── .streamlit/
│   └── config.toml             # 🎨 Streamlit theme (light, navy palette)
│
├── Data/
│   ├── X_train_cleaned.csv     # 📊 Cleaned feature matrix (training data)
│   ├── y_train_cleaned.csv     # 🎯 Target UCS values (MPa)
│   ├── materials.json          # 📚 Material property database & encodings
│   └── realistic_constraints.json  # 📏 Engineering constraint bounds
│
├── models/
│   └── checkpoints/            # 💾 5-fold cross-validated BNN checkpoints
│       ├── fold_1_best.ckpt
│       ├── fold_2_best.ckpt
│       ├── fold_3_best.ckpt
│       ├── fold_4_best.ckpt
│       └── fold_5_best.ckpt
│
├── LICENSE                     # 📝 MIT License
└── README.md                   # 📖 Documentation
```

---

## Model Architecture

The BNN is implemented in **PyTorch Lightning** and consists of:

| Component | Details |
|---|---|
| **Architecture** | Gated Residual Network (GRN) blocks with skip connections |
| **Depth** | 4 GRN blocks, hidden dim = 256 |
| **Output heads** | Dual-head: `μ` (mean strength) + `log σ` (aleatoric uncertainty) |
| **Uncertainty** | Epistemic (ensemble variance) + Aleatoric (predicted σ) |
| **Training** | 5-fold cross-validation, Huber loss, AdamW + weight decay |
| **MC Dropout** | 5 forward passes per model × 5 models = 25-sample ensemble |

### Uncertainty Quantification

```
Total uncertainty = √(Aleatoric² + Epistemic²)

Aleatoric  — irreducible noise captured by the σ-head
Epistemic  — model disagreement across ensemble folds
```

---

## Optimization Engine

The inverse design uses **SLSQP** (Sequential Least-Squares Programming) from `scipy.optimize` with:

### Decision Variables
| Variable | Bounds |
|---|---|
| Cement | 100 – 900 kg/m³ |
| SCMs (Silica Fume, Fly Ash, …) | 0 – 400 kg/m³ each |
| Sand | 500 – 1500 kg/m³ |
| Filler | 0 – 400 kg/m³ |
| Water | 100 – 250 kg/m³ |
| Superplasticizer | 5 – 50 kg/m³ |
| Fiber | 0 – (μ + 2σ) kg/m³ |

### Hard Engineering Constraints
| Constraint | Range |
|---|---|
| Total Binder | 500 – 1100 kg/m³ |
| Water/Binder | 0.14 – 0.20 |
| SP/Binder | 0.5 % – 3 % |
| Sand/Binder | 0.8 – 1.5 |
| Total Mass | 2200 – 2550 kg/m³ |

### Objective Function
```
minimize: MSE(predicted, target) + β·σ² + reliability_penalty + soft_constraints
```
where `β` is the **risk aversion** parameter:
- `β = 0.0` → Aggressive (exploratory)
- `β = 0.5` → Balanced *(default)*
- `β = 2.0` → Conservative (low uncertainty)

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git

### 1. Clone

```bash
git clone https://github.com/murad-geoAi/uhpc-inverse-design.git
cd uhpc-inverse-design
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Dashboard Usage

1. **Set Target Strength** — enter the desired UCS in MPa (e.g. 150 MPa)
2. **Choose Risk Strategy** — Aggressive / Balanced / Conservative
3. **Select Materials** — Specimen geometry, Primary binder, SCMs, Filler, Fiber, Curing regime
4. **Run Optimization** — click the button; results populate in seconds
5. **Inspect Results** — Strength distribution plot, composition donut, component ranking, and raw data table

---

## Supported Supplementary Cementitious Materials (SCMs)

Silica Fume · Fly Ash · Limestone Powder · Quartz Powder · Glass Powder · Rice Husk Ash · Metakaolin · GGBFS · Steel Slag

## License

This project is licensed under the [MIT License](LICENSE).