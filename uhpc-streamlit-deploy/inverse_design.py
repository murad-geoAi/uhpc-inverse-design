import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
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

        # Load Materials Data
        if not materials_file_path or not os.path.exists(materials_file_path):
            raise FileNotFoundError(f"Missing materials data at {materials_file_path}")

        with open(materials_file_path, "r") as f:
            self.materials_data = json.load(f)

        self.feature_columns = self.materials_data["Feature_Order"]
        self.input_dim = len(self.feature_columns)
        self.mappings = self.materials_data.get("Categorical_Mappings", {})

        # Load Scalers
        print("Refitting scalers from training data...")
        if not data_file_path or not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Missing training data at {data_file_path}")
        df_train = pd.read_csv(data_file_path)

        self.scaler_X = StandardScaler()
        self.scaler_X.fit(df_train[self.feature_columns].values)

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

        # Load Ensemble
        self.models = self._load_ensemble()
        print(f"Loaded {len(self.models)} models from {self.checkpoint_dir}")

        # Calculate Fiber Constraints from Training Data
        fiber_col = "Amount / Quantity of Fiber"
        if fiber_col in df_train.columns:
            self.fiber_mean = df_train[fiber_col].mean()
            self.fiber_std = df_train[fiber_col].std()
        else:
            self.fiber_mean = 100.0  # Fallback
            self.fiber_std = 50.0

        fiber_col_1 = "Amount / Quantity of Fiber .1"
        if fiber_col_1 in df_train.columns and df_train[fiber_col_1].count() > 0:
            self.fiber_mean_1 = df_train[fiber_col_1].mean()
            self.fiber_std_1 = df_train[fiber_col_1].std()
        else:
            self.fiber_mean_1 = 0.0
            self.fiber_std_1 = 1.0

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
        """Convert string value to integer code using loaded mappings."""
        if not value or value == "None":
            return 0  # Default or nan

        # Normalize
        norm_val = str(value).lower().strip()

        mapping = self.mappings.get(category, {})
        if norm_val in mapping:
            return float(mapping[norm_val])

        print(f"Warning: {value} not found in {category} mapping. Returning 0.")
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
        beta=0.5,  # Risk aversion parameter
        reliability_threshold=10.0,  # Max allowable sigma (soft target)
        return_diagnostics=False,
    ):
        scm_types = scm_types or []

        # --- Step 1: Material Group Definitions ---
        # Binder: Cement + Selected SCMs + Other pozzolans defined in dataset
        possible_binders = [
            "Cement",
            "Silica Fume",
            "Fly Ash",
            "Limestone Powder",
            "Quartz Powder",
            "Glass Powder",
            "Rice Husk Ash",
            "Metakaolin",
            "GGBFS",
            "Steel Slag",
        ]

        # Fluids
        fluids = ["Water", "Superplasticizer"]

        # Aggregates
        aggregates = ["Sand", "Filler"]

        # Variables to optimize
        variables = []
        bounds = []
        initial_guess = []

        # 1. Primary Binder (Cement)
        if binder_type == "Cement":
            variables.append("Cement")
            bounds.append((100.0, 900.0))
            initial_guess.append(500.0)

        # 2. SCMs (Optimized if selected)
        for scm in scm_types:
            variables.append(scm)
            bounds.append((0.0, 400.0))
            initial_guess.append(50.0)

        # 3. Aggregates
        # Sand
        variables.append("Sand")
        bounds.append((500.0, 1500.0))  # 0.8 to 1.5 Sand/Binder implied roughly
        initial_guess.append(900.0)

        # Filler (Optimized if selected)
        filler_selected = filler_type and str(filler_type).lower() not in [
            "none",
            "nan",
            "",
        ]
        if filler_selected:
            variables.append("Filler")
            bounds.append((0.0, 400.0))  # Tuned for Filler/Binder <= 0.4
            initial_guess.append(100.0)

        # 4. Fluids
        # Water
        variables.append("Water")
        bounds.append((100.0, 250.0))  # Tighter start
        initial_guess.append(160.0)

        # Superplasticizer
        variables.append("Superplasticizer")
        bounds.append((5.0, 50.0))
        initial_guess.append(15.0)

        # 5. Fibers
        fiber_selected = fiber_type and str(fiber_type).lower() not in [
            "none",
            "nan",
            "",
        ]
        fiber_limit = self.fiber_mean + 2 * self.fiber_std
        if fiber_selected:
            variables.append("Fiber")
            bounds.append((0.0, fiber_limit))
            initial_guess.append(self.fiber_mean)

        # 6. Temperature
        variables.append("Temperature")
        bounds.append((15.0, 40.0))  # Step 5: Temperature Restriction
        initial_guess.append(25.0)

        # Encode Categorical Inputs
        filler_code = self.get_categorical_code("Type of Filler", filler_type)
        fiber_code = self.get_categorical_code("Type of Fiber", fiber_type)
        curing_code = self.get_categorical_code("Curing", curing_type)
        specimen_code = self.get_categorical_code("Specimen Size", specimen_size)

        # --- Step 2: Helper Functions using x ---
        def get_var_map(x):
            return {name: val for name, val in zip(variables, x)}

        def total_binder(x):
            v = get_var_map(x)
            return sum(v.get(b, 0.0) for b in possible_binders if b in variables)

        def total_mass(x):
            v = get_var_map(x)
            # Sum of all mass components (Binder + Sand + Filler + Water + SP + Fiber)
            # Note: Temperature is not mass.
            mass = 0.0
            for name, val in v.items():
                if name != "Temperature":
                    mass += val
            return mass

        def wb_ratio(x):
            b = total_binder(x)
            if b < 1e-6:
                return 10.0  # High penalty
            return get_var_map(x).get("Water", 0.0) / b

        def sp_binder_ratio(x):
            b = total_binder(x)
            if b < 1e-6:
                return 10.0
            return (
                get_var_map(x).get("Superplasticizer", 0.0) / b
            )  # SP is kg/m3 usually, ratio is mass/mass?
            # Assuming SP is in kg/m3. Constraint is 0.5% to 3%. so 0.005 to 0.03.
            # If dataset has SP in liters or something else, adjustment might be needed.
            # Assuming mass ratio based on prompt "SP/Binder".

        def sand_binder_ratio(x):
            b = total_binder(x)
            if b < 1e-6:
                return 10.0
            return get_var_map(x).get("Sand", 0.0) / b

        def filler_binder_ratio(x):
            b = total_binder(x)
            if b < 1e-6:
                return 10.0
            v = get_var_map(x)
            return v.get("Filler", 0.0) / b

        def total_fiber(x):
            v = get_var_map(x)
            return v.get("Fiber", 0.0)

        # --- Vector Construction for Model ---
        def make_vector(x):
            v = get_var_map(x)
            row = {}

            # Helper to map variable names to columns
            # (Similar to previous implementation but using helper)
            def set_mat(name, amt):
                # Mapping logic ... (reusing reliable mapping from before)
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
                    if amt > 0:
                        for prop, val in mat_info.items():
                            if prop in self.feature_columns:
                                row[prop] = val
                    else:
                        for prop in mat_info.keys():
                            if prop in self.feature_columns:
                                row[prop] = 0.0

            # Set Binders
            for m in possible_binders:
                set_mat(m, v.get(m, 0.0))

            # Set Sand
            sand_amt = v.get("Sand", 0.0)
            if "Sand Amount  (kg/m³)" in self.feature_columns:
                row["Sand Amount  (kg/m³)"] = sand_amt
            sand_props = self.materials_data.get("Sand", {})
            for k, val in sand_props.items():
                if k in self.feature_columns:
                    row[k] = val

            # Set Fluids & Temp
            if "Water _Amount  _(kg/m³)" in self.feature_columns:
                row["Water _Amount  _(kg/m³)"] = v.get("Water", 0.0)
            if "Superplasticizer  Amount  (kg/m³)" in self.feature_columns:
                row["Superplasticizer  Amount  (kg/m³)"] = v.get(
                    "Superplasticizer", 0.0
                )
            if "Temperature (o C)" in self.feature_columns:
                row["Temperature (o C)"] = v.get("Temperature", 0.0)

            # Set Filler
            filler_amt = v.get("Filler", 0.0)
            if "Filler (kg/m³)" in self.feature_columns:
                row["Filler (kg/m³)"] = filler_amt
            if "Type of Filler" in self.feature_columns:
                row["Type of Filler"] = filler_code

            # Set Fiber
            fiber_amt = v.get("Fiber", 0.0)
            if "Amount / Quantity of Fiber" in self.feature_columns:
                row["Amount / Quantity of Fiber"] = fiber_amt
            if "Type of Fiber" in self.feature_columns:
                row["Type of Fiber"] = fiber_code

            # Fiber Props
            if fiber_selected:
                f_key = str(int(fiber_code))
                f_props = self.materials_data.get("Fiber_Properties", {}).get(f_key, {})
                if "Length (mm)" in self.feature_columns:
                    row["Length (mm)"] = f_props.get("Length (mm)", 0.0)
                if "Diameter (mm)" in self.feature_columns:
                    row["Diameter (mm)"] = f_props.get("Diameter (mm)", 0.0)
            else:
                if "Length (mm)" in self.feature_columns:
                    row["Length (mm)"] = 0.0
                if "Diameter (mm)" in self.feature_columns:
                    row["Diameter (mm)"] = 0.0

            # Categoricals
            if "Curing" in self.feature_columns:
                row["Curing"] = curing_code
            if "Specimen Size" in self.feature_columns:
                row["Specimen Size"] = specimen_code

            # Zero out unused
            for c in self.feature_columns:
                if c not in row:
                    row[c] = 0.0

            final_vec = np.array([row[c] for c in self.feature_columns])
            return final_vec

        # --- Step 4: Soft Penalties in Objective ---
        def objective(x):
            vec = make_vector(x)
            pred_mean, pred_sigma = self.predict(vec)
            pred_mean = pred_mean[0]
            pred_sigma = pred_sigma[0]

            # 1. Primary MSE Term
            mse = (pred_mean - target_strength) ** 2

            # 2. Epistemic/Aleatoric Uncertainty Penalty
            # "Risk Aversion": Penalize solutions with high uncertainty
            uncertainty_penalty = beta * (pred_sigma**2)

            # 3. Probabilistic Reliability Soft Penalty
            # If sigma exceeds 15% of mean, penalize heavily (Research Grade Safety)
            reliability_limit = 0.15 * pred_mean
            reliability_loss = 0.0
            if pred_sigma > reliability_limit:
                reliability_loss = 10.0 * (pred_sigma - reliability_limit) ** 2

            # Soft Penalties (lambda approx 0.01-0.05)
            soft_constraints_penalty = 0.0
            lam = 0.05

            # SP/Binder slightly exceeding bounds (e.g. preferred 0.5-3%, penalize if > 3% or < 0.5%?)
            # Prompt: "Add small quadratic penalties when: SP/Binder slightly exceeds bounds"
            # Hard constraints are 0.5% - 3%. This might imply soft constraints for a tighter range
            # OR ensuring the optimizer stays within hard bounds by steering it away from edges.
            # I will interpret "slightly exceeds bounds" as "if we were to relax hard constraints".
            # However, since hard constraints are active, this penalty pushes it towards the VALID region if it somehow strays,
            # or biases it effectively.
            # Let's apply penalty for being near the edge or outside if SLSQP goes slightly out.

            sp_b = sp_binder_ratio(x)
            # Target range 0.005 - 0.03
            if sp_b > 0.03:
                soft_constraints_penalty += lam * (sp_b - 0.03) ** 2
            if sp_b < 0.005:
                soft_constraints_penalty += lam * (0.005 - sp_b) ** 2

            s_b = sand_binder_ratio(x)
            # Target 0.8 - 1.5
            if s_b > 1.5:
                soft_constraints_penalty += lam * (s_b - 1.5) ** 2
            if s_b < 0.8:
                soft_constraints_penalty += lam * (0.8 - s_b) ** 2

            # Task B: Minimum SCM Soft Penalty
            if len(scm_types) > 0:
                v_map = get_var_map(x)
                total_scm_mass = sum(
                    v_map.get(s, 0.0) for s in scm_types if s in variables
                )
                b_mass = total_binder(x)
                scm_ratio = total_scm_mass / max(b_mass, 1e-6)
                min_scm_ratio = 0.05
                violation = max(0.0, min_scm_ratio - scm_ratio)
                # lambda_scm suggested 0.01-0.05
                soft_constraints_penalty += 0.05 * (violation**2)

            return (
                mse + uncertainty_penalty + reliability_loss + soft_constraints_penalty
            )

        # --- Step 3: Hard Constraints (SLSQP) ---
        # Constraints form: fun(x) >= 0
        constraints = []

        # 1. Total Binder: 500 <= Binder <= 1100
        constraints.append({"type": "ineq", "fun": lambda x: total_binder(x) - 500.0})
        constraints.append({"type": "ineq", "fun": lambda x: 1100.0 - total_binder(x)})

        # 2. Water/Binder: 0.14 <= W/B <= 0.20
        constraints.append({"type": "ineq", "fun": lambda x: wb_ratio(x) - 0.14})
        constraints.append({"type": "ineq", "fun": lambda x: 0.20 - wb_ratio(x)})

        # 3. SP/Binder: 0.5% <= SP/B <= 3% (0.005 - 0.03)
        # Note: SP in kg/m3, Binder in kg/m3. Ratio is mass/mass.
        constraints.append(
            {"type": "ineq", "fun": lambda x: sp_binder_ratio(x) - 0.005}
        )
        constraints.append({"type": "ineq", "fun": lambda x: 0.03 - sp_binder_ratio(x)})

        # 4. Fiber Limit: Amount <= Mean + 2*Std
        if fiber_selected:
            fiber_limit = self.fiber_mean + 2 * self.fiber_std
            fiber_idx = variables.index("Fiber")
            constraints.append(
                {"type": "ineq", "fun": lambda x: fiber_limit - x[fiber_idx]}
            )
            constraints.append(
                {"type": "ineq", "fun": lambda x: x[fiber_idx]}
            )  # Lower bound 0

        # 5. Sand/Binder: 0.8 <= S/B <= 1.5
        constraints.append(
            {"type": "ineq", "fun": lambda x: sand_binder_ratio(x) - 0.8}
        )
        constraints.append(
            {"type": "ineq", "fun": lambda x: 1.5 - sand_binder_ratio(x)}
        )

        # 6. Filler/Binder: F/B <= 0.40
        constraints.append(
            {"type": "ineq", "fun": lambda x: 0.40 - filler_binder_ratio(x)}
        )

        # 7. Total Mass: 2200 <= Mass <= 2550
        constraints.append({"type": "ineq", "fun": lambda x: total_mass(x) - 2200.0})
        constraints.append({"type": "ineq", "fun": lambda x: 2550.0 - total_mass(x)})

        # Prepare Optimization
        res = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"eps": 1.0, "maxiter": 300, "ftol": 1e-6},
        )

        final_amounts = res.x
        final_vector = make_vector(final_amounts)
        pred_strength, pred_unc = self.predict(final_vector)
        pred_strength = float(pred_strength[0])
        pred_unc = float(pred_unc[0])

        # Build Output
        result_mix = {}
        for i, var in enumerate(variables):
            if var != "Temperature":
                result_mix[var] = final_amounts[i]

        # Calculate Final Engineering Metrics for Diagnostics
        f_binder = total_binder(final_amounts)
        f_mass = total_mass(final_amounts)
        f_wb = wb_ratio(final_amounts)
        f_sp_b = sp_binder_ratio(final_amounts)
        f_sand_b = sand_binder_ratio(final_amounts)
        f_fiber_b = (
            0.0  # Filler/Binder was requested, adding Fiber info too just in case
        )

        # Reliability Calculation
        cv = (pred_unc / pred_strength) * 100.0 if pred_strength > 0 else 100.0
        confidence_score = max(0.0, 100.0 - cv)

        reliability_status = "High"
        if cv > 10.0:
            reliability_status = "Medium"
        if cv > 15.0 or pred_unc > reliability_threshold:
            reliability_status = "Low (Risky)"

        diagnostics = {
            "optimization_success": bool(res.success),
            "optimization_message": str(res.message),
            "target_strength": float(target_strength),
            "predicted_strength": pred_strength,
            "uncertainty": pred_unc,
            "confidence_score": confidence_score,
            "reliability_status": reliability_status,
            "engineering_props": {
                "Total Binder": f_binder,
                "Total Mass": f_mass,
                "W/B Ratio": f_wb,
                "SP/Binder": f_sp_b,
                "Sand/Binder": f_sand_b,
                "Temperature": final_amounts[variables.index("Temperature")],
            },
            "constraints_check": {
                "Binder (500-1100)": 500 <= f_binder <= 1100,
                "Mass (2200-2550)": 2200 <= f_mass <= 2550,
                "W/B (0.14-0.20)": 0.14 <= f_wb <= 0.20,
                "SP/B (0.005-0.03)": 0.005 <= f_sp_b <= 0.03,
                "Sand/B (0.8-1.5)": 0.8 <= f_sand_b <= 1.5,
                "Reliability (CV < 15%)": cv < 15.0,
            },
        }

        if return_diagnostics:
            return result_mix, pred_strength, pred_unc, diagnostics
        return result_mix, pred_strength, pred_unc


if __name__ == "__main__":
    pass
