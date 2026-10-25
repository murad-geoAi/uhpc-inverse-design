import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


def _resolve_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class GRNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x1 = F.elu(self.fc1(x))
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        gate = torch.sigmoid(self.gate(x))
        out = gate * x1
        return self.norm(residual + out)


class BayesianNN(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        depth=4,
        dropout=0.15,
        lr=7e-4,
        weight_decay=1e-4,
        huber_delta=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GRNBlock(hidden_dim, dropout) for _ in range(depth)]
        )
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)
        self.log_nu = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x = F.gelu(self.input_proj(x) + self.skip(x))
        for block in self.blocks:
            x = block(x)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x).clamp(min=-6.0, max=3.0)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma


# ==========================================
# 2. INVERSE DESIGN SYSTEM
# ==========================================
class InverseDesignSystem:
    def __init__(
        self,
        checkpoint_dir="models/checkpoints",
        materials_file="data/materials.json",
        data_file="data/X_train_cleaned.csv",
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = os.path.dirname(self.base_dir)

        local_data_dir = os.path.join(self.base_dir, "Data")
        root_data_dir = os.path.join(self.repo_root, "Data")

        self.checkpoint_dir = _resolve_existing_path(
            os.path.join(self.base_dir, "models", "checkpoints"),
            os.path.join(self.repo_root, "models", "checkpoints"),
            checkpoint_dir,
            os.path.join(self.base_dir, checkpoint_dir),
            os.path.join(self.repo_root, checkpoint_dir),
        )
        materials_file_path = _resolve_existing_path(
            os.path.join(local_data_dir, "materials.json"),
            os.path.join(root_data_dir, "materials.json"),
            materials_file,
            os.path.join(self.base_dir, materials_file),
            os.path.join(self.repo_root, materials_file),
        )
        data_file_path = _resolve_existing_path(
            os.path.join(local_data_dir, "X_train_cleaned.csv"),
            os.path.join(root_data_dir, "X_train_cleaned.csv"),
            data_file,
            os.path.join(self.base_dir, data_file),
            os.path.join(self.repo_root, data_file),
        )

        # --- 1. Load Materials Metadata ---
        if not materials_file_path or not os.path.exists(materials_file_path):
            raise FileNotFoundError(f"Missing materials data at {materials_file_path}")

        with open(materials_file_path, "r") as f:
            self.materials_data = json.load(f)

        self.feature_columns = self.materials_data["Feature_Order"]
        self.input_dim = len(self.feature_columns)
        self.mappings = self.materials_data.get("Categorical_Mappings", {})

        # --- 2. Load Scalers (Prioritize Assets over CSVs) ---
        assets_dir = os.path.join(self.base_dir, "models", "assets")
        scaler_x_path = os.path.join(assets_dir, "scaler_x.joblib")
        scaler_y_path = os.path.join(assets_dir, "scaler_y.joblib")
        constants_path = os.path.join(assets_dir, "constants.json")

        if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
            print("Loading pre-fitted scalers from assets...")
            self.scaler_X = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)
        else:
            print("Refitting scalers from raw training data...")
            if not data_file_path or not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Missing training data at {data_file_path}")
            df_train_x = pd.read_csv(data_file_path)
            self.scaler_X = StandardScaler()
            self.scaler_X.fit(df_train_x[self.feature_columns].values)

            y_path = _resolve_existing_path(
                os.path.join(os.path.dirname(data_file_path), "y_train_cleaned.csv"),
                os.path.join(local_data_dir, "y_train_cleaned.csv"),
                os.path.join(root_data_dir, "y_train_cleaned.csv"),
            )
            if not y_path or not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing training targets at {y_path}")
            df_y = pd.read_csv(y_path)
            self.scaler_y = StandardScaler()
            self.scaler_y.fit(df_y.values)

        # --- 3. Load Stats/Constants ---
        if os.path.exists(constants_path):
            with open(constants_path, 'r') as f:
                constants = json.load(f)
            self.fiber_mean = constants.get("fiber_mean", 100.0)
            self.fiber_std = constants.get("fiber_std", 50.0)
            self.fiber_mean_1 = constants.get("fiber_mean_1", 0.0)
            self.fiber_std_1 = constants.get("fiber_std_1", 1.0)
        else:
            # Fallback calculation if constants file is missing
            self.fiber_mean = 100.0
            self.fiber_std = 50.0
            self.fiber_mean_1 = 0.0
            self.fiber_std_1 = 1.0

        # --- 4. Load Model Ensemble ---
        self.models = self._load_ensemble()
        print(f"Loaded {len(self.models)} models from {self.checkpoint_dir}")

    def _load_ensemble(self):
        models = []
        if not os.path.exists(self.checkpoint_dir):
            return models
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith(".ckpt"):
                path = os.path.join(self.checkpoint_dir, f)
                try:
                    model = BayesianNN.load_from_checkpoint(
                        path, input_dim=self.input_dim
                    )
                    model.eval()
                    model.to(self.device)
                    models.append(model)
                except Exception as e:
                    print(f"Failed to load {f}: {e}")
        return models

    def predict(self, input_vector):
        if input_vector.ndim == 1:
            input_vector = input_vector.reshape(1, -1)

        x_scaled = self.scaler_X.transform(input_vector)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        means = []
        sigmas = []

        with torch.no_grad():
            for model in self.models:
                for _ in range(5):
                    mu, sigma = model(x_tensor)
                    means.append(mu)
                    sigmas.append(sigma)

        stacked_means = torch.stack(means).cpu().numpy()
        stacked_sigmas = torch.stack(sigmas).cpu().numpy()

        pred_mean_scaled = np.mean(stacked_means, axis=0)
        aleatoric = np.mean(stacked_sigmas**2, axis=0)
        epistemic = np.var(stacked_means, axis=0)
        total_unc_scaled = np.sqrt(aleatoric + epistemic)

        pred_mean = self.scaler_y.inverse_transform(pred_mean_scaled)
        scale_factor = self.scaler_y.scale_[0]
        final_unc = total_unc_scaled * scale_factor

        return pred_mean.flatten(), final_unc.flatten()

    def get_categorical_code(self, category, value):
        if not value or value == "None":
            return 0.0
        norm_val = str(value).lower().strip()
        mapping = self.mappings.get(category, {})
        if norm_val in mapping:
            return float(mapping[norm_val])
        return 0.0

    def optimize_mix(
        self,
        target_strength,
        binder_type,
        scm_types,
        filler_type,
        fiber_type,
        curing_type,
        specimen_size,
        beta=0.5,
        reliability_threshold=10.0,
        return_diagnostics=False,
    ):
        scm_types = scm_types or []
        possible_binders = [
            "Cement", "Silica Fume", "Fly Ash", "Limestone Powder", 
            "Quartz Powder", "Glass Powder", "Rice Husk Ash", 
            "Metakaolin", "GGBFS", "Steel Slag"
        ]

        variables = []
        bounds = []
        initial_guess = []

        # 1. Primary Binder (Cement)
        if binder_type == "Cement":
            variables.append("Cement")
            bounds.append((100.0, 900.0))
            initial_guess.append(500.0)

        # 2. SCMs
        for scm in scm_types:
            variables.append(scm)
            bounds.append((0.0, 400.0))
            initial_guess.append(50.0)

        # 3. Aggregates
        variables.append("Sand")
        bounds.append((500.0, 1500.0))
        initial_guess.append(900.0)

        filler_selected = filler_type and str(filler_type).lower() not in ["none", "nan", ""]
        if filler_selected:
            variables.append("Filler")
            bounds.append((0.0, 400.0))
            initial_guess.append(100.0)

        # 4. Fluids
        variables.append("Water")
        bounds.append((100.0, 250.0))
        initial_guess.append(160.0)

        variables.append("Superplasticizer")
        bounds.append((5.0, 50.0))
        initial_guess.append(15.0)

        # 5. Fibers
        fiber_selected = fiber_type and str(fiber_type).lower() not in ["none", "nan", ""]
        if fiber_selected:
            fiber_limit = self.fiber_mean + 2 * self.fiber_std
            variables.append("Fiber")
            bounds.append((0.0, fiber_limit))
            initial_guess.append(self.fiber_mean)

        # 6. Temperature
        variables.append("Temperature")
        bounds.append((15.0, 40.0))
        initial_guess.append(25.0)

        # Encode Categorical Inputs
        filler_code = self.get_categorical_code("Type of Filler", filler_type)
        fiber_code = self.get_categorical_code("Type of Fiber", fiber_type)
        curing_code = self.get_categorical_code("Curing", curing_type)
        specimen_code = self.get_categorical_code("Specimen Size", specimen_size)

        def get_var_map(x):
            return {name: val for name, val in zip(variables, x)}

        def total_binder(x):
            v = get_var_map(x)
            return sum(v.get(b, 0.0) for b in possible_binders if b in variables)

        def total_mass(x):
            v = get_var_map(x)
            mass = sum(val for name, val in v.items() if name != "Temperature")
            return mass

        def wb_ratio(x):
            b = total_binder(x)
            return get_var_map(x).get("Water", 0.0) / max(b, 1e-6)

        def sp_binder_ratio(x):
            b = total_binder(x)
            return get_var_map(x).get("Superplasticizer", 0.0) / max(b, 1e-6)

        def sand_binder_ratio(x):
            b = total_binder(x)
            return get_var_map(x).get("Sand", 0.0) / max(b, 1e-6)

        def filler_binder_ratio(x):
            b = total_binder(x)
            return get_var_map(x).get("Filler", 0.0) / max(b, 1e-6)

        def make_vector(x):
            v = get_var_map(x)
            row = {}

            def set_mat(name, amt):
                col_map = {
                    "Cement": "Cement Amount  (kg/m³)",
                    "Silica Fume": "Silica Fume (kg/m³)",
                    "Fly Ash": "Flayash Amount   (kg/m³)",
                    "Limestone Powder": "Limestone Powder  (kg/m3)",
                    "Quartz Powder": "Quartzpowder (kg/m3)",
                    "Glass Powder": "Glass powder (kg/m3)",
                    "Rice Husk Ash": "Rice husk ash (kg/m3)",
                    "Metakaolin": "Metakaolin (kg/m³)",
                    "GGBFS": "GGBFS  (kg/m³)",
                    "Steel Slag": "Slag Amount   (kg/m³)",
                }
                amt_col = col_map.get(name)
                if amt_col and amt_col in self.feature_columns:
                    row[amt_col] = amt
                    mat_info = self.materials_data.get(name, {})
                    for prop, val in mat_info.items():
                        if prop in self.feature_columns:
                            row[prop] = val if amt > 0 else 0.0

            for m in possible_binders: set_mat(m, v.get(m, 0.0))
            
            sand_amt = v.get("Sand", 0.0)
            if "Sand Amount  (kg/m³)" in self.feature_columns: row["Sand Amount  (kg/m³)"] = sand_amt
            for k, val in self.materials_data.get("Sand", {}).items():
                if k in self.feature_columns: row[k] = val

            if "Water _Amount  _(kg/m³)" in self.feature_columns: row["Water _Amount  _(kg/m³)"] = v.get("Water", 0.0)
            if "Superplasticizer  Amount  (kg/m³)" in self.feature_columns: row["Superplasticizer  Amount  (kg/m³)"] = v.get("Superplasticizer", 0.0)
            if "Temperature (o C)" in self.feature_columns: row["Temperature (o C)"] = v.get("Temperature", 0.0)
            if "Filler (kg/m³)" in self.feature_columns: row["Filler (kg/m³)"] = v.get("Filler", 0.0)
            if "Type of Filler" in self.feature_columns: row["Type of Filler"] = filler_code
            if "Amount / Quantity of Fiber" in self.feature_columns: row["Amount / Quantity of Fiber"] = v.get("Fiber", 0.0)
            if "Type of Fiber" in self.feature_columns: row["Type of Fiber"] = fiber_code

            if fiber_selected:
                f_props = self.materials_data.get("Fiber_Properties", {}).get(str(int(fiber_code)), {})
                if "Length (mm)" in self.feature_columns: row["Length (mm)"] = f_props.get("Length (mm)", 0.0)
                if "Diameter (mm)" in self.feature_columns: row["Diameter (mm)"] = f_props.get("Diameter (mm)", 0.0)
            else:
                if "Length (mm)" in self.feature_columns: row["Length (mm)"] = 0.0
                if "Diameter (mm)" in self.feature_columns: row["Diameter (mm)"] = 0.0

            if "Curing" in self.feature_columns: row["Curing"] = curing_code
            if "Specimen Size" in self.feature_columns: row["Specimen Size"] = specimen_code

            for c in self.feature_columns:
                if c not in row: row[c] = 0.0
            return np.array([row[c] for c in self.feature_columns])

        def objective(x):
            vec = make_vector(x)
            pred_mean, pred_sigma = self.predict(vec)
            pred_mean, pred_sigma = pred_mean[0], pred_sigma[0]
            
            mse = (pred_mean - target_strength) ** 2
            uncertainty_penalty = beta * (pred_sigma**2)
            
            reliability_limit = 0.15 * pred_mean
            reliability_loss = 10.0 * (pred_sigma - reliability_limit) ** 2 if pred_sigma > reliability_limit else 0.0
            
            soft_penalties = 0.0
            lam = 0.05
            sp_b = sp_binder_ratio(x)
            if sp_b > 0.03: soft_penalties += lam * (sp_b - 0.03) ** 2
            if sp_b < 0.005: soft_penalties += lam * (0.005 - sp_b) ** 2
            
            s_b = sand_binder_ratio(x)
            if s_b > 1.5: soft_penalties += lam * (s_b - 1.5) ** 2
            if s_b < 0.8: soft_penalties += lam * (0.8 - s_b) ** 2

            return mse + uncertainty_penalty + reliability_loss + soft_penalties

        constraints = [
            {"type": "ineq", "fun": lambda x: total_binder(x) - 500.0},
            {"type": "ineq", "fun": lambda x: 1100.0 - total_binder(x)},
            {"type": "ineq", "fun": lambda x: wb_ratio(x) - 0.14},
            {"type": "ineq", "fun": lambda x: 0.20 - wb_ratio(x)},
            {"type": "ineq", "fun": lambda x: sp_binder_ratio(x) - 0.005},
            {"type": "ineq", "fun": lambda x: 0.03 - sp_binder_ratio(x)},
            {"type": "ineq", "fun": lambda x: sand_binder_ratio(x) - 0.8},
            {"type": "ineq", "fun": lambda x: 1.5 - sand_binder_ratio(x)},
            {"type": "ineq", "fun": lambda x: 0.40 - filler_binder_ratio(x)},
            {"type": "ineq", "fun": lambda x: total_mass(x) - 2200.0},
            {"type": "ineq", "fun": lambda x: 2550.0 - total_mass(x)},
        ]

        res = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 300, "ftol": 1e-6})

        final_amounts = res.x
        final_vector = make_vector(final_amounts)
        pred_strength, pred_unc = self.predict(final_vector)
        pred_strength, pred_unc = float(pred_strength[0]), float(pred_unc[0])

        result_mix = {var: final_amounts[i] for i, var in enumerate(variables) if var != "Temperature"}
        
        cv = (pred_unc / pred_strength) * 100.0 if pred_strength > 0 else 100.0
        reliability_status = "High" if cv <= 10.0 else "Medium" if cv <= 15.0 else "Low (Risky)"

        diagnostics = {
            "optimization_success": bool(res.success),
            "predicted_strength": pred_strength,
            "uncertainty": pred_unc,
            "confidence_score": max(0.0, 100.0 - cv),
            "reliability_status": reliability_status,
            "engineering_props": {
                "Total Binder": total_binder(final_amounts),
                "Total Mass": total_mass(final_amounts),
                "W/B Ratio": wb_ratio(final_amounts),
                "Temperature": final_amounts[variables.index("Temperature")],
            }
        }

        return (result_mix, pred_strength, pred_unc, diagnostics) if return_diagnostics else (result_mix, pred_strength, pred_unc)


if __name__ == "__main__":
    pass
